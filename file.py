import os
import shutil


base_dir = r"C:\Users\zappe\Desktop\창학\SuCo-HRNE"
src_dir = os.path.join(base_dir, "data", "Sample", "01.원천데이터")
dst_query = os.path.join(base_dir, "data", "query")
dst_ref = os.path.join(base_dir, "data", "reference")

os.makedirs(dst_query, exist_ok=True)
os.makedirs(dst_ref, exist_ok=True)

for filename in os.listdir(src_dir):
    if not filename.endswith(".wav"):
        continue

    if "_Org.wav" in filename:
        shutil.copy(os.path.join(src_dir, filename), os.path.join(dst_query, filename))
    else:
        shutil.copy(os.path.join(src_dir, filename), os.path.join(dst_ref, filename))

print("파일 정리가 완료되었습니다.")