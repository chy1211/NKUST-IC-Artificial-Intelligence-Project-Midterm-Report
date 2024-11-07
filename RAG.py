import json
import pdfplumber
import requests
import faiss
import numpy as np

# 打開 PDF 文件並提取文本
pdf = pdfplumber.open("HANDBOOK FOR FACULTY.pdf")
text = "".join(page.extract_text() for page in pdf.pages)


# 將文本分段處理
def split_text(text, max_length=500):
    paragraphs = text.split('\n')
    chunks, current_chunk = [], ""
    for paragraph in paragraphs:
        if len(current_chunk) + len(paragraph) > max_length:
            chunks.append(current_chunk)
            current_chunk = paragraph
        else:
            current_chunk += " " + paragraph
    if current_chunk:
        chunks.append(current_chunk)
    return chunks


text_chunks = split_text(text)

# 構建 API 請求
url = "http://localhost:1234/v1/embeddings"
headers = {"Content-Type": "application/json"}

vectors = []
for chunk in text_chunks:
    data = {"model": "text-embedding-multilingual-e5-large-instruct", "input": chunk}
    response = requests.post(url, headers=headers, data=json.dumps(data))
    if response.status_code == 200:
        embeddings = response.json()["data"]
        vectors.extend(item['embedding'] for item in embeddings)
    else:
        exit(1)

# 檢查 embeddings 的結構
if isinstance(vectors, list) and all(isinstance(item, list) for item in vectors):
    vectors = np.array(vectors).astype('float32')
    dimension = vectors.shape[1]
    index_flat = faiss.IndexFlatL2(dimension)
    index_flat.add(vectors)
    faiss.write_index(index_flat, "embeddings.index")
else:
    exit(1)

# 讀取 question.json
with open("question.json", "r", encoding="utf-8") as f:
    questions = json.load(f)["questions"]

expected_dimension = 1024
results = []

# 對每個問題進行相似性搜尋
for question in questions:
    question_data = {"model": "text-embedding-multilingual-e5-large-instruct", "input": question}
    question_response = requests.post(url, headers=headers, data=json.dumps(question_data))

    if question_response.status_code == 200:
        data = question_response.json().get('data')
        if data and len(data) > 0:
            embedding = data[0].get('embedding')
            if embedding and len(embedding) == expected_dimension:
                embedding = np.array(embedding).astype('float32').reshape(1, -1)
                distances, indices = index_flat.search(embedding, 5)
                nearest_texts = "\n".join([f"[{i+1}] {text_chunks[idx]}" for i, idx in enumerate(indices[0])])

                llm_url = "http://localhost:1234/v1/chat/completions"
                llm_headers = {"Content-Type": "application/json"}
                llm_data = {
                    "model": "taidetaide-lx-7b-chat",
                    "messages": [
                        {"role": "system", "content": f"根據人類與 AI 助理之間的對話以及一份來源列表，為 AI 助理撰寫最終答案。請遵循以下指引：\n\n1. 如果人類詢問到以下四種情境之一，如沒有則忽略此指引：\n   - 緊急傷病\n   - 心理諮詢與輔導\n   - 性別平等或校園性侵事件\n   - 交通安全或事故\n   - 吸食毒品等違法行為\n\n   則先提供對應的聯絡方式，然後再回答原本的問題。聯絡方式如下：\n   1. 緊急傷病：\n      - 校安中心：0800-550995\n      - 衛保組第一校區分機：31250-31255\n      - 衛保組建工校區分機：12530-12534\n      - 衛保組燕巢校區分機：18535\n      - 衛保組楠梓校區分機：22086-22089\n      - 衛保組旗津校區分機：25085\n\n   2. 心理諮詢與輔導：\n      - 生活關懷師：\n        - 第一校區分機：31224\n        - 建工校區分機：13405\n        - 燕巢校區分機：18656\n        - 楠梓校區分機：22065\n        - 旗津校區分機：25035\n      - 個別諮商與輔導：請先至 https://stu.nkust.edu.tw/p/412-1007-5474.php 上網預約初談。\n\n   3. 性別平等或校園性侵事件：\n      - 應24小時內聯絡學校性平會：\n        - 建工/燕巢校區： (07) 3814526#12501，sml@nkust.edu.tw\n        - 第一/旗津校區： (07) 3617141#31290/12502，ping1005@nkust.edu.tw\n        - 楠梓校區： (07) 6011000#31232，annysung@nkust.edu.tw\n\n   4. 交通安全或事故：\n      - 校安中心：0800-550995\n\n   5. 吸食毒品等違法行為：\n      - 校安中心：0800-550995\n\n2. 回答人類的問題，並確保你提及來源中的所有相關細節，儘可能使用與來源中完全相同的詞彙。\n3. 答案必須僅基於來源，不得引入任何額外信息。\n4. 所有數字，如價格、日期、時間或電話號碼，必須與來源中顯示的一模一樣。\n5. 根據來源提供盡可能全面的答案。包含所有重要的細節，以及任何適用的附加條件或限制。\n6. 答案必須用繁體中文撰寫。\n7. 不要嘗試捏造答案：如果答案無法從來源中找到，你應該承認自己不知道，並回答 \"NOT_ENOUGH_INFORMATION\"。\n8. 請根據語義自然換行，確保排版清晰。\n9. 請使用與查詢相同的語言回應。\n10. 一定要列點說明。\n11. 回答請盡可能簡潔有力，並控制在100字內。\n12. 完全避免使用「根據」、「來源」。\n13. 請避免產生不連貫或不完整的句子，要保持句子的完整性和邏輯性。\n14.請不要一直重複輸出同樣的句子。\n你將在開始前獲得幾個範例:範例 1：\n來源：\n[1] <產品或服務> 信息頁面\n是的，<公司> 提供各種選擇或變化的 <產品或服務>。\n\n人類：你們賣 <產品或服務> 嗎？\nAI：是的，<公司> 賣 <產品或服務>。還有其他需要幫助的嗎？\n\n範例 2：\n來源：\n[1] Andrea - 維基百科\nAndrea 是一個全球通用的名字，適用於男性和女性。\n\n人類：天氣如何？\nAI：NOT_ENOUGH_INFORMATION\n\n開始吧！我們一步一步來，確保我們得出正確的答案。"},
                        {"role": "user", "content": f"來源：{nearest_texts}\n 人類： {question}"}
                    ],
                    "temperature": 0.7,
                    "max_tokens": -1,
                    "stream": False
                }

                llm_response = requests.post(llm_url, headers=llm_headers, data=json.dumps(llm_data))
                llm_response_json = llm_response.json()

                results.append({
                    "model": llm_data["model"],
                    "question": question,
                    "response": llm_response_json["choices"][0]["message"]["content"]
                })
                print(f"question:{question}")
                print(f"response:{llm_response_json['choices'][0]['message']['content']}")
            else:
                print(f"Invalid embedding for question: {question}")
                print(f"Expected dimension: {expected_dimension}")
                print(f"Actual dimension: {len(embedding)}")
                print(f"Embedding: {embedding}")

# 將所有結果儲存為 JSON 檔案
with open("results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=4)
