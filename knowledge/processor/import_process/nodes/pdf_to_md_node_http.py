import os
import shutil
import time
import zipfile
from pathlib import Path
from typing import Tuple

import requests

from knowledge.processor.import_process.base import BaseNode, setup_logging
from knowledge.processor.import_process.exceptions import (
    ConfigurationError,
    FileProcessingError,
    PdfConversionError,
    ValidationError,
)
from knowledge.processor.import_process.state import ImportGraphState


class PdfToMdNode(BaseNode):
    """Convert PDF to Markdown with the MinerU HTTP API."""

    name = "pdf_to_md_node"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_url = os.getenv("MINERU_BASE_URL", "https://mineru.net/api/v4").rstrip("/")
        self.api_token = os.getenv("MINERU_BASE_API_TOKEN", "").strip()
        self.poll_interval_seconds = int(os.getenv("MINERU_POLL_INTERVAL_SECONDS", "3"))
        self.timeout_seconds = int(os.getenv("MINERU_TIMEOUT_SECONDS", "600"))
        self.http_timeout_seconds = int(os.getenv("MINERU_HTTP_TIMEOUT_SECONDS", "60"))

    def process(self, state: ImportGraphState) -> ImportGraphState:
        pdf_path, output_dir = self._validate_state_inputs_path(state)
        zip_url = self._upload_and_poll(pdf_path)
        md_path = self._download_and_extract(zip_url, output_dir, pdf_path.stem)

        state["md_path"] = md_path
        try:
            state["md_content"] = Path(md_path).read_text(encoding="utf-8")
        except OSError as exc:
            self.logger.warning(f"Failed to read markdown content: {exc}")

        return state

    def _validate_state_inputs_path(self, state: ImportGraphState) -> Tuple[Path, Path]:
        self.log_step("step1", "validate pdf path and output dir")

        import_file_path = state.get("import_file_path") or state.get("pdf_path") or ""
        file_dir = state.get("file_dir") or ""
        if not import_file_path:
            raise ValidationError("Missing PDF path in state", self.name)

        pdf_path = Path(import_file_path)
        if not pdf_path.exists():
            raise FileProcessingError(f"PDF file does not exist: {pdf_path}", self.name)
        if pdf_path.suffix.lower() != ".pdf":
            raise ValidationError(f"Only PDF files are supported: {pdf_path.name}", self.name)

        output_dir = Path(file_dir) if file_dir else pdf_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"PDF path: {pdf_path}")
        self.logger.info(f"Output dir: {output_dir}")
        return pdf_path, output_dir

    def _require_api_token(self) -> str:
        if not self.api_token:
            raise ConfigurationError("Missing env var MINERU_BASE_API_TOKEN", self.name)
        return self.api_token

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self._require_api_token()}",
            "Content-Type": "application/json",
        }

    def _upload_and_poll(self, pdf_path: Path) -> str:
        self.log_step("step2", "request MinerU http api")

        file_name = pdf_path.name
        payload = {"files": [{"name": file_name}]}
        upload_url = f"{self.base_url}/file-urls/batch"

        try:
            response = requests.post(
                upload_url,
                headers=self._headers(),
                json=payload,
                timeout=self.http_timeout_seconds,
            )
        except requests.RequestException as exc:
            raise PdfConversionError(f"Failed to get upload url: {exc}", self.name) from exc

        resp_data = self._parse_api_response(response, "Failed to get upload url")
        file_urls = resp_data.get("file_urls") or []
        batch_id = resp_data.get("batch_id")
        if not file_urls or not batch_id:
            raise PdfConversionError(
                f"MinerU response missing file_urls or batch_id: {resp_data}",
                self.name,
            )

        signed_url = file_urls[0]
        self._upload_pdf_file(signed_url, pdf_path)
        return self._poll_result(batch_id, file_name)

    def _upload_pdf_file(self, signed_url: str, pdf_path: Path) -> None:
        self.logger.info("Uploading PDF to MinerU")
        try:
            file_data = pdf_path.read_bytes()
        except OSError as exc:
            raise FileProcessingError(f"Failed to read PDF: {exc}", self.name) from exc

        session = requests.Session()
        session.trust_env = False

        try:
            put_response = session.put(
                signed_url,
                data=file_data,
                timeout=self.http_timeout_seconds,
            )
            if put_response.status_code != 200:
                put_response = session.put(
                    signed_url,
                    data=file_data,
                    headers={"Content-Type": "application/pdf"},
                    timeout=self.http_timeout_seconds,
                )
            if put_response.status_code != 200:
                raise PdfConversionError(
                    f"Failed to upload PDF: {put_response.status_code} {put_response.text}",
                    self.name,
                )
        except requests.RequestException as exc:
            raise PdfConversionError(f"Network error while uploading PDF: {exc}", self.name) from exc
        finally:
            session.close()

        self.logger.info("PDF upload completed")

    def _poll_result(self, batch_id: str, file_name: str) -> str:
        poll_url = f"{self.base_url}/extract-results/batch/{batch_id}"
        start_time = time.time()

        while True:
            if time.time() - start_time > self.timeout_seconds:
                raise PdfConversionError(
                    f"MinerU conversion timed out after {self.timeout_seconds} seconds",
                    self.name,
                )

            try:
                response = requests.get(
                    poll_url,
                    headers=self._headers(),
                    timeout=self.http_timeout_seconds,
                )
            except requests.RequestException as exc:
                self.logger.warning(f"Polling failed, retry later: {exc}")
                time.sleep(self.poll_interval_seconds)
                continue

            if 500 <= response.status_code < 600:
                self.logger.warning(f"MinerU busy ({response.status_code}), retry later")
                time.sleep(self.poll_interval_seconds)
                continue

            resp_data = self._parse_api_response(response, "Failed to poll result")
            extract_results = resp_data.get("extract_result") or []
            if not extract_results:
                self.logger.info("MinerU result not ready yet")
                time.sleep(self.poll_interval_seconds)
                continue

            result_item = extract_results[0]
            state_status = result_item.get("state", "")
            if state_status == "done":
                full_zip_url = result_item.get("full_zip_url")
                if not full_zip_url:
                    raise PdfConversionError("MinerU finished without a result url", self.name)
                self.logger.info(f"MinerU finished for {file_name}")
                return full_zip_url

            if state_status == "failed":
                err_msg = result_item.get("err_msg") or "unknown error"
                raise PdfConversionError(f"MinerU failed: {err_msg}", self.name)

            self.logger.info(f"MinerU state: {state_status}")
            time.sleep(self.poll_interval_seconds)

    def _parse_api_response(self, response: requests.Response, action: str) -> dict:
        if response.status_code != 200:
            raise PdfConversionError(f"{action}: {response.status_code} {response.text}", self.name)

        try:
            body = response.json()
        except ValueError as exc:
            raise PdfConversionError(f"{action}: invalid json response", self.name) from exc

        if body.get("code") != 0:
            raise PdfConversionError(f"{action}: {body}", self.name)

        data = body.get("data")
        if not isinstance(data, dict):
            raise PdfConversionError(f"{action}: missing data field", self.name)
        return data

    def _download_and_extract(self, zip_url: str, output_dir: Path, pdf_stem: str) -> str:
        self.log_step("step3", "download and extract MinerU result")

        try:
            response = requests.get(zip_url, timeout=self.http_timeout_seconds)
        except requests.RequestException as exc:
            raise PdfConversionError(f"Failed to download result zip: {exc}", self.name) from exc

        if response.status_code != 200:
            raise PdfConversionError(
                f"Failed to download result zip: {response.status_code} {response.text}",
                self.name,
            )

        zip_path = output_dir / f"{pdf_stem}_result.zip"
        extract_dir = output_dir / pdf_stem

        try:
            zip_path.write_bytes(response.content)
            if extract_dir.exists():
                shutil.rmtree(extract_dir)
            extract_dir.mkdir(parents=True, exist_ok=True)

            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(extract_dir)
        except (OSError, zipfile.BadZipFile) as exc:
            raise PdfConversionError(f"Failed to process result zip: {exc}", self.name) from exc

        md_path = self._locate_markdown_file(extract_dir, pdf_stem)
        self.logger.info(f"Markdown path: {md_path}")
        return str(md_path)

    def _locate_markdown_file(self, extract_dir: Path, pdf_stem: str) -> Path:
        expected_path = extract_dir / "hybrid_auto" / f"{pdf_stem}.md"
        if expected_path.exists():
            return expected_path

        md_files = list(extract_dir.rglob("*.md"))
        if not md_files:
            raise PdfConversionError("No markdown file found after extraction", self.name)

        matched = self._find_best_md_candidate(md_files, pdf_stem)
        if matched.stem != pdf_stem:
            renamed = matched.with_name(f"{pdf_stem}.md")
            try:
                matched.rename(renamed)
                matched = renamed
            except OSError as exc:
                self.logger.warning(f"Failed to rename markdown file: {exc}")
        return matched

    @staticmethod
    def _find_best_md_candidate(md_files: list[Path], pdf_stem: str) -> Path:
        for md_file in md_files:
            if md_file.stem == pdf_stem:
                return md_file
        for md_file in md_files:
            if md_file.name.lower() == "full.md":
                return md_file
        return md_files[0]


if __name__ == "__main__":
    setup_logging()
    pdf_to_md_node = PdfToMdNode()

    init_state = {
        "import_file_path": r"D:\develop\develop\workspace\pycharm\251020\shopkeeper_brain\knowledge\processor\import_process\import_temp_dir\test.pdf",
        "file_dir": r"D:\develop\develop\workspace\pycharm\251020\shopkeeper_brain\knowledge\processor\import_process\import_temp_dir",
    }
    print(pdf_to_md_node.process(init_state))
