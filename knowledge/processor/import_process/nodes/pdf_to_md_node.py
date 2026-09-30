import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Tuple

from knowledge.processor.import_process.base import BaseNode, setup_logging
from knowledge.processor.import_process.exceptions import (
    FileProcessingError,
    PdfConversionError,
    ValidationError,
)
from knowledge.processor.import_process.state import ImportGraphState


class PdfToMdNode(BaseNode):
    """
    PDF 转 Markdown 节点。
    """

    name = "pdf_to_md_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        """
        执行 PDF 转 Markdown。
        """
        import_file_path, file_dir_path = self._validate_state_inputs_path(state)

        processed_code = self._execute_mineru(import_file_path, file_dir_path)
        if processed_code != 0:
            raise PdfConversionError("MinerU解析PDF失败", self.name)

        md_path = self._get_md_paths(import_file_path, file_dir_path)
        md_path_obj = Path(md_path)
        if not md_path_obj.exists():
            raise PdfConversionError(
                f"MinerU执行完成但未生成md文件: {md_path_obj}",
                self.name,
            )

        state["md_path"] = md_path
        return state

    def _validate_state_inputs_path(self, state: ImportGraphState) -> Tuple[Path, Path]:
        """
        校验输入路径。
        """
        self.log_step("step1", "对状态中的路径参数进行校验")

        import_file_path = state.get("import_file_path", "")
        file_dir = state.get("file_dir", "")

        if not import_file_path:
            raise ValidationError("待解析的文件不存在", self.name)

        import_file_path_obj = Path(import_file_path)
        if not import_file_path_obj.exists():
            raise FileProcessingError("待解析的文件路径不存在", self.name)

        if not file_dir:
            file_dir = import_file_path_obj.parent

        file_dir_path_obj = Path(file_dir)
        self.logger.info(f"上传文件的路径: {import_file_path_obj}")
        self.logger.info(f"输出目录: {file_dir_path_obj}")

        return import_file_path_obj, file_dir_path_obj

    def _execute_mineru(self, import_file_path: Path, file_dir_path: Path) -> int:
        """
        调用 MinerU 执行转换。
        """
        self.log_step("step2", "执行MinerU解析PDF")

        cmd = [
            sys.executable,
            "-m",
            "mineru.cli.client",
            "-p",
            str(import_file_path),
            "-o",
            str(file_dir_path),
            "--source",
            "local",
        ]

        process_start_time = time.time()
        proc = subprocess.Popen(
            args=cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            errors="replace",
            text=True,
            encoding="utf-8",
            bufsize=1,
        )

        combined_output_lines = []
        if proc.stdout is not None:
            for line in proc.stdout:
                combined_output_lines.append(line)
                self.logger.info(f"执行MinerU产生的日志：{line.rstrip()}")

        processed_code = proc.wait()
        process_end_time = time.time()
        combined_output = "".join(combined_output_lines)

        if "No module named 'shapely'" in combined_output:
            self.logger.error("MinerU缺少运行依赖 shapely，请先安装 shapely 后再重试")
            return 1

        if processed_code == 0:
            self.logger.info(
                f"MinerU成功解析PDF文件：{import_file_path.name} 耗时:{process_end_time - process_start_time:.2f}s"
            )
        else:
            self.logger.error(f"MinerU解析PDF文件失败：{import_file_path.name}")

        return processed_code

    def _get_md_paths(self, import_file_path: Path, file_dir_path: Path) -> str:
        file_name = import_file_path.stem
        md_path = file_dir_path / file_name / "hybrid_auto" / f"{file_name}.md"
        return str(md_path)


if __name__ == "__main__":
    setup_logging()
    pdf_to_md_node = PdfToMdNode()

    pdf_to_md_node_init_state = {
        "import_file_path": r"D:\develop\develop\workspace\pycharm\251020\shopkeeper_brain\knowledge\processor\import_process\import_temp_dir\万用表的使用.pdf",
        "file_dir": r"D:\develop\develop\workspace\pycharm\251020\shopkeeper_brain\knowledge\processor\import_process\import_temp_dir",
    }
    processed_result = pdf_to_md_node.process(pdf_to_md_node_init_state)

    print(json.dumps(processed_result, indent=4, ensure_ascii=False))
