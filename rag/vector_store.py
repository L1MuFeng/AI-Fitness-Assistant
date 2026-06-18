import os
import sys
from pathlib import Path

# 直接运行本文件时，将项目根目录加入模块搜索路径
_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from model.factory import embed_model
from utils.config_handler import chroma_conf
from utils.file_handler import (
    get_file_md5_hex,
    listdir_with_allowed_type,
    pdf_loader,
    txt_loader,
)
from utils.logger_handler import logger
from utils.path_tool import get_abs_path


class VectorStoreService:
    def __init__(self):
        self.vector_store = Chroma(
            collection_name=chroma_conf["collection_name"],
            embedding_function=embed_model,
            persist_directory=get_abs_path(chroma_conf["persist_directory"]),
        )
        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size=chroma_conf["chunk_size"],
            chunk_overlap=chroma_conf["chunk_overlap"],
            separators=chroma_conf["separators"],
            length_function=len,
        )
    
    def get_retriever(self):
        return self.vector_store.as_retriever(search_kwargs={"k": chroma_conf["k"]})
    
    def load_document(self) -> dict:
        loaded: list[str] = []
        skipped: list[str] = []
        failed: list[str] = []

        def check_md5_hex(md5_for_check: str) -> bool:
            if not md5_for_check:
                return False
            md5_path = get_abs_path(chroma_conf["md5_hex_store"])
            if not os.path.exists(md5_path):
                open(md5_path, "w", encoding="utf-8").close()
                return False

            with open(md5_path, "r", encoding="utf-8") as f:
                for line in f.readlines():
                    if line.strip() == md5_for_check:
                        return True
            return False

        def save_md5_hex(md5_for_check: str) -> None:
            with open(get_abs_path(chroma_conf["md5_hex_store"]), "a", encoding="utf-8") as f:
                f.write(md5_for_check + "\n")

        def get_file_documents(read_path: str):
            if read_path.endswith(".txt"):
                return txt_loader(read_path)
            if read_path.endswith(".pdf"):
                return pdf_loader(read_path)
            return []

        allowed_files_path: list[str] = listdir_with_allowed_type(
            chroma_conf["data_path"],
            tuple(chroma_conf["allow_knowledge_file_type"]),
        )

        if not allowed_files_path:
            logger.warning("[加载知识库] data 目录下未找到 txt/pdf 文档")

        for path in allowed_files_path:
            md5_hex = get_file_md5_hex(path)
            if check_md5_hex(md5_hex):
                logger.info(f"[加载知识库]{path}内容已存在知识库内，跳过")
                skipped.append(path)
                continue
            try:
                documents: list[Document] = get_file_documents(path)
                if not documents:
                    logger.warning(f"[加载知识库]{path}内容为空，跳过")
                    failed.append(path)
                    continue

                split_document: list[Document] = self.spliter.split_documents(documents)
                if not split_document:
                    logger.warning(f"[加载知识库]{path}分片后没有有效内容，跳过")
                    failed.append(path)
                    continue

                self.vector_store.add_documents(split_document)
                save_md5_hex(md5_hex)
                logger.info(f"[加载知识库]{path}内容加载成功")
                loaded.append(path)
            except Exception as e:
                logger.error(f"[加载知识库]{path}内容加载失败: {str(e)}", exc_info=True)
                failed.append(path)

        return {"loaded": loaded, "skipped": skipped, "failed": failed}


def ensure_knowledge_base_loaded() -> dict:
    """检查并增量同步 data/ 下的知识库文档（已索引文件通过 MD5 跳过）。"""
    logger.info("[加载知识库]开始检查知识库...")
    service = VectorStoreService()
    result = service.load_document()
    logger.info(
        "[加载知识库]同步完成：新增 %d，跳过 %d，失败 %d",
        len(result["loaded"]),
        len(result["skipped"]),
        len(result["failed"]),
    )
    return result


if __name__ == "__main__":
    sync_result = ensure_knowledge_base_loaded()
    print(sync_result)

    vs = VectorStoreService()
    retriever = vs.get_retriever()
    res = retriever.invoke("迷路")
    for r in res:
        print(r.page_content)
        print("-" * 20)