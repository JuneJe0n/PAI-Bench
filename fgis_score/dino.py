import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModel
from tqdm import tqdm

ATTRIBUTES = [
    'skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g',
    'l_ear', 'r_ear', 'ear_r', 'nose', 'mouth', 'u_lip',
    'l_lip', 'neck', 'neck_l', 'cloth', 'hair', 'hat'
]


def extract_region_embedding(original_image, binary_mask, processor, model, device):
    """
    Binary mask를 사용하여 원본 이미지에서 해당 영역의 DINO 임베딩 추출
    
    Args:
        original_image: PIL Image (원본 이미지)
        binary_mask: PIL Image (binary mask)
        processor: DINO image processor
        model: DINO model
        device: torch device
    
    Returns:
        numpy array: DINO embedding (768-dim for dinov2-base)
    """
    # convert image and mask into numpy array
    img_array = np.array(original_image)
    mask_array = np.array(binary_mask)
    
    binary = mask_array > 127
    if not binary.any():
        return None
    
    # 바운딩 박스 찾기
    coords = np.argwhere(binary)
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    
    # 마스크 적용된 영역 추출
    masked_image = img_array.copy()
    masked_image[~binary] = 255  # 배경을 흰색으로
    
    # 바운딩 박스로 크롭
    cropped = masked_image[y_min:y_max+1, x_min:x_max+1]
    
    # PIL Image로 변환
    pil_cropped = Image.fromarray(cropped)
    # path = "./data/pil_cropped.jpg"
    # pil_cropped.save("./data/pil_cropped.jpg")
    
    # DINO 임베딩 추출
    inputs = processor(images=pil_cropped, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state[:, 0].cpu().numpy()[0]
    
    return embedding


def extract_embeddings_for_celeb(celeb_name, images_dir, masks_dir, angle, processor, model, device):
    """
    특정 celeb의 모든 이미지에 대해 face region별 임베딩 추출
    
    Args:
        celeb_name: 연예인 이름 (e.g., "BrunoMars")
        images_dir: 원본 이미지 디렉토리 (e.g., "./assets/images/BrunoMars")
        masks_dir: Binary mask 디렉토리 (e.g., "./assets/binary_mask_output/BrunoMars")
        processor: DINO processor
        model: DINO model
        device: torch device
    
    Returns:
        DataFrame: 각 행은 이미지, 각 열은 face region의 임베딩
    """
    results = []
    
    # 이미지 파일 리스트 가져오기
    image_files = sorted([f for f in os.listdir(images_dir) # ./assets/images/BrunoMars 내의 모든 이미지 파일들
                         if f.endswith(('.jpg', '.png', '.jpeg', 'JPG'))])

    print(f"\n🎭 Processing {celeb_name}: {len(image_files)} images")
    
    for image_file in tqdm(image_files, desc=f"Extracting embeddings"):
        # 파일명에서 숫자 추출 (e.g., "0.jpg" → "0")
        image_number = os.path.splitext(image_file)[0] # for cc 
        
        # 원본 이미지 로드
        image_path = os.path.join(images_dir, image_file)
        original_image = Image.open(image_path).convert('RGB')
        
        # 해당 이미지의 마스크 디렉토리
        mask_image_dir = os.path.join(masks_dir, image_number) # ./assets/binary_mask_output/BrunoMars/1
        # mask_image_dir = os.path.join(masks_dir, image_name) # ./assets/cc_binary_mask_output/BrunoMars/angle_type/darkened_image_1
        mask_files = sorted([f for f in os.listdir(mask_image_dir) if f.endswith((".jpg", ".png", ".jpeg"))])
        
        if not os.path.exists(mask_image_dir):
            print(f"⚠️  Mask directory not found: {mask_image_dir}")
            continue
        
        # 각 region별 임베딩 저장
        row_data = {
            'celeb': celeb_name,
            'image_id': image_number
        }
        
        # row_data = {
        #     'celeb': celeb_name,
        #     'angle': angle, # same angle or diff angle
        #     'image_name': image_name, # darkened_image_1
        #     # 'image_id': image_number # same_angle, diff_angle
        #     'image_id': "0" # 0
        # }
        
        # 각 face region에 대해 처리
        for mask_file in mask_files:
            mask_path = os.path.join(mask_image_dir, mask_file) # 전체 path
            number = os.path.splitext(mask_file)[0].split('_')[-1]
            
            # region이 없으면 (모두 0이면) 0 벡터, 아니면 임베딩 추출 #
            
            if os.path.exists(mask_path):
                # Binary mask 로드
                binary_mask = Image.open(mask_path).convert('L')
                binary_mask_array = np.array(binary_mask)
                if np.all(binary_mask_array < 127): 
                    # 해당 region이 없으면 None으로 설정 후, 나중에 유사도 계산할 때 필터링하기
                    row_data[number] = None
                else:
                    # 임베딩 추출
                    embedding = extract_region_embedding(
                        original_image, 
                        binary_mask, 
                        processor, 
                        model, 
                        device
                    )
                    row_data[number] = embedding
        
        results.append(row_data)
    
    # DataFrame 생성
    df = pd.DataFrame(results)
    
    return df


def extract_all_embeddings(base_images_dir, base_masks_dir, output_path="face_embeddings.pkl", angle=""):
    """
    모든 연예인에 대해 임베딩 추출 및 저장
    
    Args:
        base_images_dir: 이미지 베이스 디렉토리 (e.g., "./assets/images")
        base_masks_dir: 마스크 베이스 디렉토리 (e.g., "./assets/binary_mask_output")
        output_path: 저장할 파일 경로
    """
    # DINO 모델 로드
    print("🔄 Loading DINO model...")
    processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
    model = AutoModel.from_pretrained('facebook/dinov2-base')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    print(f"✅ Model loaded on {device}")
    
    # 모든 연예인 리스트
    celebs = [d for d in os.listdir(base_images_dir) 
             if os.path.isdir(os.path.join(base_images_dir, d))]
    
    print(f"\n📋 Found {len(celebs)} celebrities: {celebs}")
    
    # 모든 데이터를 담을 리스트
    all_dataframes = []
    
    # 각 연예인별로 처리
    for celeb in celebs:
        # images_dir = os.path.join(base_images_dir, celeb, "cropped", angle) # ./assets/images/BrunoMars/cropped/angle_type
        # masks_dir = os.path.join(base_masks_dir, celeb, angle) # ./assets/binary_mask_output/BrunoMars/angle_type
        
        # images_dir = os.path.join(base_images_dir, celeb, "cropped") # ./assets/images/BrunoMars/cropped
        # masks_dir = os.path.join(base_masks_dir, celeb) # ./assets/binary_mask_output/BrunoMars
        
        images_dir = os.path.join(base_images_dir, celeb) # ./assets/color_changed_images/BrunoMars
        masks_dir = os.path.join(base_masks_dir, celeb) # ./assets/cc_binary_mask_output/BrunoMars
        
        if not os.path.exists(masks_dir):
            print(f"⚠️  Skipping {celeb}: masks directory not found")
            continue
        
        # 임베딩 추출
        df = extract_embeddings_for_celeb(
            celeb, 
            images_dir, 
            masks_dir, 
            angle,
            processor, 
            model, 
            device
        )
        
        all_dataframes.append(df)
        print(f"✅ {celeb}: {len(df)} images processed")
    
    # 모든 DataFrame 합치기
    final_df = pd.concat(all_dataframes, ignore_index=True)
    
    print(f"\n📊 Total DataFrame shape: {final_df.shape}")
    print(f"Columns: {final_df.columns.tolist()}")
    
    # 저장
    final_df.to_pickle(output_path)
    print(f"💾 Saved to {output_path}")
    
    # CSV로도 저장 (임베딩 제외, 메타데이터만)
    meta_df = final_df[['celeb', 'angle', 'image_name', 'image_id']].copy()
    csv_path = output_path.replace('.pkl', '_meta.csv')
    meta_df.to_csv(csv_path, index=False)
    print(f"💾 Metadata saved to {csv_path}")
    
    return final_df


# ============= 사용 예시 =============

if __name__ == "__main__":
    # 경로 설정 
    # original 
    base_images_dir = "./assets/images"
    base_masks_dir = "./assets/binary_mask_output"
    output_path = f"./embedding/face_region_embeddings.pkl"

    # change color
    # angle = "0"
    # base_images_dir = "./assets/color_changed_images"
    # base_masks_dir = "./assets/cc_binary_mask_output"
    # output_path = f"./embedding/cc_face_region_embeddings_{angle}.pkl"
    
    # 임베딩 추출 및 저장
    df = extract_all_embeddings(
        base_images_dir, 
        base_masks_dir, 
        output_path
    )
    
    print("\n" + "="*50)
    print("✨ Extraction Complete!")
    print("="*50)

    # DataFrame 정보 출력
    print(f"\nDataFrame Info:")
    print(f"  - Shape: {df.shape}")
    print(f"  - Celebrities: {df['celeb'].unique().tolist()}")
    print(f"  - Total images: {len(df)}")
