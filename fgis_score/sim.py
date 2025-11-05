import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# 데이터 로드
with open('./embedding/face_region_embeddings.pkl', 'rb') as f: # original image
    embeddings = pickle.load(f)

# DataFrame으로 변환
df = pd.DataFrame(embeddings)

reference_faces = {
    'BrunoMars': '000001.jpg',
    'Dicaprio': '000016.jpg',
    'FanBingbing': '000034.jpg',
    'IshiharaSatomi': '000049.jpg',
    'Jennie': '000099.jpg',
    'JKRowling': '000015.jpg',
    'Obama': '000008.jpg',
    'SoonjaeLee': '000086_0.jpg',
    'TaylorSwift': '000033.jpg',
    'TomHolland': '000001.jpg'
}

# feature embedding region 컬럼 추출 (1, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13)
# '1', '10', '11' 등의 숫자 컬럼들
feature_cols = [col for col in df.columns if col not in ['celeb', 'image_id']]

# 유사도 계산 결과를 저장할 리스트
results = []

# 각 celeb에 대해 유사도 계산
for celeb_name, reference_image_id in reference_faces.items():
    # 해당 celeb의 모든 이미지 필터링
    df_celeb = df[df['celeb'] == celeb_name].copy()
    
    # 기준 이미지 찾기
    df_reference = df_celeb[df_celeb['image_id'] == reference_image_id]
    
    if len(df_reference) == 0:
        print(f"⚠️ Warning: Reference image {reference_image_id} not found for {celeb_name}")
        continue
    
    reference_row = df_reference.iloc[0]
    
    # 기준 이미지를 제외한 나머지 이미지들
    df_compare = df_celeb[df_celeb['image_id'] != reference_image_id]
    
    # 각 비교 이미지에 대해 유사도 계산
    celeb_similarities = []
    
    for idx, compare_row in df_compare.iterrows():
        # 각 feature별 코사인 유사도 계산
        feature_similarities = []
        
        for feature_col in feature_cols:
            # 기준 임베딩의 해당 feature
            emb_reference = reference_row[feature_col]
            # 비교 임베딩의 해당 feature
            emb_compare = compare_row[feature_col]
            
            # 둘 다 None이 아닌 경우에만 계산
            if emb_reference is not None and emb_compare is not None:
                if isinstance(emb_reference, (list, np.ndarray)) and isinstance(emb_compare, (list, np.ndarray)):
                    emb_reference = np.array(emb_reference).reshape(1, -1)
                    emb_compare = np.array(emb_compare).reshape(1, -1)
                    similarity = cosine_similarity(emb_reference, emb_compare)[0][0]
                    feature_similarities.append(similarity)
        
        # 이미지 간 평균 유사도 계산
        if len(feature_similarities) > 0:
            avg_similarity = np.mean(feature_similarities)
            celeb_similarities.append(avg_similarity)
            
            # 개별 결과 저장
            results.append({
                'celeb': celeb_name,
                'reference_image_id': reference_image_id,
                'compare_image_id': compare_row['image_id'],
                'cosine_similarity': avg_similarity,
                'num_features_compared': len(feature_similarities)
            })
    
    # celeb별 평균 유사도 계산
    if len(celeb_similarities) > 0:
        celeb_avg_similarity = np.mean(celeb_similarities)
        print(f"✅ {celeb_name}: 평균 유사도 = {celeb_avg_similarity:.6f} (비교 이미지 수: {len(celeb_similarities)})")
    else:
        print(f"⚠️ {celeb_name}: 유사도를 계산할 이미지가 없습니다.")

# 결과를 DataFrame으로 변환하고 저장
df_results = pd.DataFrame(results)
df_results.to_csv('./similarity_results.csv', index=False)

print(f"\n{'='*60}")
print(f"✅ 총 {len(results)}개의 유사도가 계산되었습니다.")
print(f"📊 DataFrame shape: {df_results.shape}")

# celeb별 평균 유사도 집계
print(f"\n{'='*60}")
print("📊 Celeb별 평균 유사도:")
print(f"{'='*60}")
celeb_summary = df_results.groupby('celeb').agg({
    'cosine_similarity': ['mean', 'std', 'count']
}).round(6)
print(celeb_summary)

# 전체 평균
overall_avg = df_results['cosine_similarity'].mean()
print(f"\n{'='*60}")
print(f"🎯 전체 평균 유사도: {overall_avg:.6f}")
print(f"{'='*60}")

# 5. 각 celeb에 대해 유사도 계산
# for idx, row_0 in df_0.iterrows():
#     celeb_name = row_0['celeb']
#     angle_0 = row_0['angle']
#     image_id_0 = row_0['image_id']
    
#     # diff angle에서 같은 celeb 찾기
#     df_diff_celeb = df_diff[df_diff['celeb'] == celeb_name]
    
#     for idx_diff, row_diff in df_diff_celeb.iterrows():
#         # 각 feature별 코사인 유사도 계산
#         feature_similarities = []
        
#         for feature_col in feature_cols:
#             # 0 임베딩의 해당 feature
#             emb_0 = row_0[feature_col]
#             # diff 임베딩의 해당 feature
#             emb_diff = row_diff[feature_col]
            
#             if emb_0 is not None and emb_diff is not None: # 둘 중 하나라도 None이면 계산 안함
#                 # 코사인 유사도 계산 (1차원 벡터를 2차원으로 reshape)
#                 print(f"diff_feature_col: {feature_col}")
#                 if isinstance(emb_0, (list, np.ndarray)) and isinstance(emb_diff, (list, np.ndarray)):
#                     emb_0 = np.array(emb_0).reshape(1, -1)
#                     emb_diff = np.array(emb_diff).reshape(1, -1)
#                     similarity = cosine_similarity(emb_0, emb_diff)[0][0]
#                     feature_similarities.append(similarity)
        
#         print("\n")
#         # 평균 유사도 계산
#         avg_similarity = np.mean(feature_similarities)
        
#         results.append({
#             'celeb': celeb_name,
#             'base_angle': angle_0,
#             'base_image_id': image_id_0,
#             'compare_type': 'diff',
#             'compare_angle': row_diff['angle'],
#             'compare_image_id': row_diff['image_id'],
#             'cosine_similarity': avg_similarity
#         })
    
#     # same angle에서 같은 celeb 찾기
#     df_same_celeb = df_same[df_same['celeb'] == celeb_name]
    
#     for idx_same, row_same in df_same_celeb.iterrows():
#         # 각 feature별 코사인 유사도 계산
#         feature_similarities = []
        
#         for feature_col in feature_cols:
#             # 0 임베딩의 해당 feature
#             emb_0 = row_0[feature_col]
#             # same 임베딩의 해당 feature
#             emb_same = row_same[feature_col]
            
#             if emb_0 is not None and emb_same is not None: # 둘 중 하나라도 None이면 계산 안함
#                 # 코사인 유사도 계산
#                 print(f"same_feature_col: {feature_col}")
#                 if isinstance(emb_0, (list, np.ndarray)) and isinstance(emb_same, (list, np.ndarray)):
#                     emb_0 = np.array(emb_0).reshape(1, -1)
#                     emb_same = np.array(emb_same).reshape(1, -1)
#                     similarity = cosine_similarity(emb_0, emb_same)[0][0]
#                     feature_similarities.append(similarity)
#         print("\n")
        
#         # 평균 유사도 계산
#         avg_similarity = np.mean(feature_similarities)
        
#         results.append({
#             'celeb': celeb_name,
#             'base_angle': angle_0,
#             'base_image_id': image_id_0,
#             'compare_type': 'same',
#             'compare_angle': row_same['angle'],
#             'compare_image_id': row_same['image_id'],
#             'cosine_similarity': avg_similarity
#         })

# # 6. 결과를 DataFrame으로 변환하고 저장
# df_results = pd.DataFrame(results)
# df_results.to_csv('./similarity.csv', index=False)

# print(f"✅ 총 {len(results)}개의 유사도가 계산되었습니다.")
# print(f"📊 DataFrame shape: {df_results.shape}")
# print("\n첫 10개 결과:")
# print(df_results.head(10))
# print("\n통계:")
# print(df_results.groupby(['celeb', 'compare_type'])['cosine_similarity'].agg(['mean', 'std', 'count']))

# diff_avg = np.mean(df_results[df_results['compare_type'] == "diff"]['cosine_similarity'])
# same_avg = np.mean(df_results[df_results['compare_type'] == "same"]['cosine_similarity'])
# print(f"\ndiff avg: {diff_avg:.6f}")
# print(f"same_avg: {same_avg:.6f}")