import mimetypes
import os

# 尝试导入第三方库
try:
    import magic
except ImportError:
    magic = None

try:
    import chardet
except ImportError:
    chardet = None


def has_bom(data):
    """检查字节数据是否包含常见文本文件的BOM（字节顺序标记）。"""
    boms = [
        b"\xff\xfe\x00\x00",  # UTF-32 LE
        b"\x00\x00\xfe\xff",  # UTF-32 BE
        b"\xff\xfe",  # UTF-16 LE
        b"\xfe\xff",  # UTF-16 BE
        b"\xef\xbb\xbf",  # UTF-8
    ]
    return any(data.startswith(bom) for bom in boms)


def is_text_file(filepath):
    """
    判断给定文件是否为文本文件。
    返回: True如果可能是文本文件，否则返回False。
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"文件 {filepath} 不存在或不是文件")

    # 存储各方法的检查结果
    checks = {
        "extension": False,
        "mimetype_guess": False,
        "python_magic_mime": False,
        "has_null_bytes": False,
        "content_decode": False,
        "chardet": False,
    }

    # 方法1：检查文件扩展名
    text_extensions = [
        ".txt",
        ".csv",
        ".json",
        ".xml",
        ".log",
        ".md",
        ".html",
        ".htm",
        ".js",
        ".css",
        ".py",
        ".java",
        ".c",
        ".cpp",
        ".h",
        ".tsv",
        ".yml",
        ".yaml",
        ".ini",
        ".cfg",
        ".conf",
    ]
    ext = os.path.splitext(filepath)[1].lower()
    checks["extension"] = ext in text_extensions

    # 方法2：使用mimetypes猜测MIME类型
    mime_type, _ = mimetypes.guess_type(filepath)
    if mime_type:
        checks["mimetype_guess"] = mime_type.startswith("text/")

    # 方法3：使用python-magic检测MIME类型
    if magic:
        try:
            mime = magic.from_file(filepath, mime=True)
            checks["python_magic_mime"] = mime.startswith("text/")
        except:
            pass

    # 方法4：检查空字节（排除有BOM的情况）
    try:
        with open(filepath, "rb") as f:
            content = f.read(1024)
            if not has_bom(content):
                checks["has_null_bytes"] = b"\x00" in content
    except:
        pass

    # 方法5：尝试解码内容
    decode_success = False
    encodings_to_try = ["utf-8", "iso-8859-1", "cp1252"]
    if chardet:
        # 如果chardet可用，优先使用检测到的编码
        try:
            with open(filepath, "rb") as f:
                raw_data = f.read(1024)
                result = chardet.detect(raw_data)
                if result["confidence"] > 0.7 and result["encoding"]:
                    encodings_to_try.insert(0, result["encoding"])
        except:
            pass

    for enc in encodings_to_try:
        try:
            with open(filepath, "r", encoding=enc, errors="strict") as f:
                f.read(1024)
                decode_success = True
                break
        except (UnicodeDecodeError, LookupError):
            continue
        except:
            break
    checks["content_decode"] = decode_success

    # 方法6：使用chardet检测编码置信度
    if chardet:
        try:
            with open(filepath, "rb") as f:
                raw_data = f.read(1024)
                result = chardet.detect(raw_data)
                checks["chardet"] = (
                    result["confidence"] > 0.7 and result["encoding"] is not None
                )
        except:
            pass

    # 综合判断逻辑
    if checks["has_null_bytes"]:
        return False  # 存在空字节且无BOM，很可能是二进制文件

    if checks["python_magic_mime"]:
        return True  # 专业库检测为文本类型

    if checks["content_decode"] and (checks["extension"] or checks["mimetype_guess"]):
        return True  # 解码成功且符合其他文本特征

    if checks["chardet"]:
        return True  # 检测到高置信度的编码

    if checks["mimetype_guess"]:
        return True  # 根据扩展名猜测的MIME类型为文本

    if checks["extension"]:
        return True  # 扩展名在已知文本类型列表中

    return False  # 所有检查均未通过
