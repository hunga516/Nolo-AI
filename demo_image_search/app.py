import os
import gradio as gr
import spaces
from PIL import Image
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


IMAGES_DIR = "images"

# Cấu hình thư mục chứa ảnh. Bạn đã chuyển ~6930 ảnh vào thư mục images/ ở cùng cấp app.py.

# Global state
model = None
index = None
image_paths = None

def auto_fix_paths_on_startup():
    """Đảm bảo các đường dẫn trong vector.index.paths trỏ tới thư mục images/ ở project root."""
    paths_file = "vector.index.paths"

    if not os.path.exists(paths_file):
        print("❌ Không tìm thấy file vector.index.paths")
        return False

    with open(paths_file, 'r') as f:
        paths = [line.strip() for line in f]

    def to_images_dir(path: str) -> str:
        filename = os.path.basename(path)
        return os.path.join(IMAGES_DIR, filename)

    # Kiểm tra nhanh xem một vài dòng đầu có nằm trong images/ chưa
    sample = paths[:5]
    needs_fixing = any(
        (
            '/content/drive/' in p or '/Users/' in p or 'C:\\' in p or
            not p.replace('\\', '/').startswith('images/')
        )
        for p in sample if p
    )

    if needs_fixing:
        print("🔧 Chuẩn hóa đường dẫn ảnh về thư mục images/ ...")
        new_paths = [to_images_dir(p) for p in paths]
        with open(paths_file, 'w') as f:
            for p in new_paths:
                f.write(p + '\n')
        print(f"✅ Đã cập nhật {len(new_paths)} đường dẫn ảnh về images/")
        return True

    print("✅ Các đường dẫn ảnh đã trỏ đúng tới thư mục images/")
    return False

def resolve_image_path(path: str) -> str:
    """Chuyển đường dẫn đã lưu thành đường dẫn thực tế, mặc định trong images/."""
    if not path:
        return path
    if os.path.isabs(path):
        return path
    # Nếu đường dẫn tồn tại (tương đối so với project), dùng luôn
    if os.path.exists(path):
        return path
    # Ngược lại, giả định file nằm trong images/
    return os.path.join(IMAGES_DIR, os.path.basename(path))

def load_model_and_index():
    """Tải model CLIP, FAISS index và chuẩn hóa đường dẫn ảnh."""
    global model, index, image_paths
    
    # Chuẩn hóa đường dẫn trong vector.index.paths nếu cần
    auto_fix_paths_on_startup()
    
    # Tải model CLIP (SentenceTransformer)
    model = SentenceTransformer('clip-ViT-B-32')
    
    # Tải FAISS index
    index_path = "vector.index"
    if os.path.exists(index_path):
        index = faiss.read_index(index_path)
        with open(index_path + '.paths', 'r') as f:
            raw_paths = [line.strip() for line in f]
        image_paths = [resolve_image_path(p) for p in raw_paths]
        print(f"📦 Đã tải index với {len(image_paths)} ảnh")

        # Kiểm tra nhanh 5 ảnh đầu có tồn tại không
        existing_images = [p for p in image_paths[:5] if os.path.exists(p)]
        print(f"✅ Xác minh {len(existing_images)}/5 ảnh mẫu tồn tại trong {IMAGES_DIR}/")
        
    else:
        raise FileNotFoundError("Không tìm thấy FAISS index. Vui lòng cung cấp vector.index và vector.index.paths")

@spaces.GPU
def retrieve_similar_images_gpu(query_input, query_type, top_k=5):
    """
    Tìm ảnh tương tự sử dụng GPU.
    query_input: chuỗi văn bản hoặc ảnh PIL
    query_type: 'text' hoặc 'image'
    """
    global model, index, image_paths
    
    if model is None or index is None:
        load_model_and_index()
    
    # Mã hóa truy vấn theo loại (văn bản hoặc ảnh)
    if query_type == 'text':
        if not query_input or query_input.strip() == '':
            return [], "Vui lòng nhập nội dung tìm kiếm (văn bản)"
        query_features = model.encode(query_input.strip())
    else:  # image
        if query_input is None:
            return [], "Vui lòng tải lên một ảnh"
        query_features = model.encode(query_input)
    
    # Định dạng vector để FAISS có thể tìm kiếm
    query_features = query_features.astype(np.float32).reshape(1, -1)
    
    # Tìm kiếm ảnh tương tự trong FAISS index
    distances, indices = index.search(query_features, min(top_k, len(image_paths)))
    
    # Thu thập kết quả ảnh trả về
    retrieved_images = []
    info_text = f"Tìm thấy {len(indices[0])} ảnh tương tự:\n"
    
    for i, idx in enumerate(indices[0]):
        if idx < len(image_paths):
            img_path = resolve_image_path(image_paths[int(idx)])
            if os.path.exists(img_path):
                try:
                    img = Image.open(img_path)
                    # Chuyển sang RGB nếu cần
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    retrieved_images.append(img)
                    info_text += f"{i+1}. Khoảng cách: {distances[0][i]:.4f} | Tệp: {os.path.basename(img_path)}\n"
                except Exception as e:
                    print(f"Lỗi khi mở ảnh {img_path}: {e}")
            else:
                print(f"Không tìm thấy ảnh: {img_path}")
    
    if not retrieved_images:
        info_text = "❌ Không tìm thấy ảnh. Hãy đảm bảo các tệp nằm trong thư mục images/."
    
    return retrieved_images, info_text

def search_by_text(text_query, top_k):
    """Tìm kiếm bằng văn bản."""
    if not text_query or text_query.strip() == '':
        return [], "Vui lòng nhập nội dung tìm kiếm (văn bản)"
    
    try:
        images, info = retrieve_similar_images_gpu(text_query, 'text', top_k)
        return images, info
    except Exception as e:
        return [], f"❌ Lỗi khi tìm kiếm bằng văn bản: {str(e)}"

def search_by_image(image_query, top_k):
    """Tìm kiếm bằng ảnh tải lên."""
    if image_query is None:
        return [], "Vui lòng tải lên một ảnh"
    
    try:
        images, info = retrieve_similar_images_gpu(image_query, 'image', top_k)
        return images, info
    except Exception as e:
        return [], f"❌ Lỗi khi tìm kiếm bằng ảnh: {str(e)}"



    
# Tải model và index khi khởi động ứng dụng
try:
    load_model_and_index()
    startup_message = f"✅ Đã tải model và index thành công! Cơ sở dữ liệu có {len(image_paths) if image_paths else 0} ảnh."
except Exception as e:
    startup_message = f"❌ Lỗi khi tải model hoặc index: {str(e)}"
    print(f"Lỗi khởi động: {e}")

# Tạo giao diện Gradio
with gr.Blocks(title="Tìm kiếm ảnh tương tự", theme=gr.themes.Soft()) as app:
    gr.Markdown("# 🔍 Tìm kiếm ảnh tương tự")
    gr.Markdown("Tìm ảnh giống nhau bằng mô tả văn bản hoặc tải ảnh lên")
    gr.Markdown(f"**Trạng thái:** {startup_message}")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 📝 Tìm kiếm bằng văn bản")
            text_input = gr.Textbox(
                label="Nhập nội dung cần tìm",
                placeholder="ví dụ: 'một chiếc xe màu đỏ', 'hoàng hôn trên núi', 'người với chú chó'",
                lines=2
            )
            text_top_k = gr.Slider(
                minimum=1, maximum=10, value=5, step=1,
                label="Số lượng kết quả"
            )
            text_search_btn = gr.Button("🔍 Tìm bằng văn bản", variant="primary")
        
        with gr.Column():
            gr.Markdown("### 🖼️ Tìm kiếm bằng ảnh")
            image_input = gr.Image(
                label="Tải ảnh để tìm ảnh tương tự",
                type="pil"
            )
            image_top_k = gr.Slider(
                minimum=1, maximum=10, value=5, step=1,
                label="Số lượng kết quả"
            )
            image_search_btn = gr.Button("🔍 Tìm bằng ảnh", variant="primary")
    
    with gr.Row():
        with gr.Column():
            info_output = gr.Textbox(
                label="Thông tin tìm kiếm",
                lines=4,
                interactive=False
            )
        
    with gr.Row():
        results_gallery = gr.Gallery(
            label="Ảnh tương tự",
            show_label=True,
            elem_id="gallery",
            columns=3,
            rows=2,
            object_fit="contain",
            height="auto"
        )
    
    # Event handlers
    text_search_btn.click(
        fn=search_by_text,
        inputs=[text_input, text_top_k],
        outputs=[results_gallery, info_output],
        show_progress=True
    )
    
    image_search_btn.click(
        fn=search_by_image,
        inputs=[image_input, image_top_k],
        outputs=[results_gallery, info_output],
        show_progress=True
    )
    
    # Cho phép nhấn Enter để tìm kiếm văn bản
    text_input.submit(
        fn=search_by_text,
        inputs=[text_input, text_top_k],
        outputs=[results_gallery, info_output],
        show_progress=True
    )

if __name__ == "__main__":
    app.launch()
