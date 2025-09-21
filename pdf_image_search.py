import fitz  # PyMuPDF for PDF parsing
import io
from PIL import Image
import torch
import clip
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

# Step 1: Load PDF and extract images
def extract_images_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    images = []
    for page_num in range(len(doc)):
        for img_index, img in enumerate(doc[page_num].get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            images.append((f"page{page_num}_img{img_index}", pil_image))
    return images

# Step 2: Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Step 3: Encode images
def encode_images(images):
    image_embeddings = {}
    for img_id, pil_img in images:
        img_tensor = preprocess(pil_img).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = model.encode_image(img_tensor)
        image_embeddings[img_id] = embedding / embedding.norm(dim=-1, keepdim=True)
    return image_embeddings

# Step 4: Encode text query
def encode_text(query):
    text_tokens = clip.tokenize([query]).to(device)
    with torch.no_grad():
        text_embedding = model.encode_text(text_tokens)
    return text_embedding / text_embedding.norm(dim=-1, keepdim=True)

# Step 5: Similarity search
def find_best_match(image_embeddings, query_embedding):
    best_score, best_img_id = -1, None
    for img_id, img_embedding in image_embeddings.items():
        similarity = (query_embedding @ img_embedding.T).item()
        if similarity > best_score:
            best_score, best_img_id = similarity, img_id
    return best_img_id, best_score

# === Example usage ===
if __name__ == "__main__":
    pdf_path = "sample.pdf"
    query = "a diagram of a neural network"

    # Extract and encode
    images = extract_images_from_pdf(pdf_path)
    image_embeddings = encode_images(images)
    query_embedding = encode_text(query)

    # Find best match
    best_img_id, score = find_best_match(image_embeddings, query_embedding)

    print(f"Best match: {best_img_id} (score={score:.4f})")
    # Display result
    for img_id, img in images:
        if img_id == best_img_id:
            img.show()
