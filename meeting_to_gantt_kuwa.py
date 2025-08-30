import os
import base64
# from openai import OpenAI

# from PIL import Image
from io import BytesIO
import random


meeting_text = """
李協理：
好的，時間到了。我們今天討論的重點是這篇來自ICLR 2025的論文《Efficient Action-Constrained RL via ARAM》。我們要討論其架構是否適用於我們目前的AGV路徑規劃與臂控系統。我們會有三位同仁各自報告5分鐘，接著兩位成員提出問題。那我們就請陳工開始。

第一位報告人：陳資深工程師（14:00-14:05）

陳工程師：
這篇論文的核心是「ARAM」，也就是「Acceptance-Rejection with Augmented MDP」的簡稱。傳統在行為受限的強化學習中，要不是依賴QP求解、就是用生成模型做後處理。而ARAM引入一個更輕量的機制，從無約束策略中先採樣，再檢查是否合法行為，若不合法則reject。這種accept-reject策略實作起來其實非常適合我們在產線上的小型移動機器人，尤其是那些要遵守轉角、速度或負載限制的模組。

更進一步，為了提高accept rate，他們提出augmented MDP：若違反constraint，會在狀態轉移中加一個self-loop並給予懲罰，這樣policy自然會學會避開那些常被reject的行為。整體結構可以套用在我們現有的SAC或DDPG架構上，只需改改buffer結構跟reward設計即可。

提問一：黃助理工程師
那陳哥，我有點疑問，我們目前的控制是離散動作空間，這個accept-reject方法會不會太慢？會不會採樣一堆無效動作導致inference latency變長？

陳工程師：
這點其實論文有提到，在他們的Reacher和HalfCheetah實驗中，ARAM推論時間是所有方法最短的，原因是它幾乎不用QP。再者，如果是離散動作，只需要判斷合法與否，不像連續動作還要計算機率密度，比想像中快。

提問二：李協理
那這個懲罰設計會不會導致我們policy太保守？例如擔心違規而不敢探索？

陳工程師：
他們用了multi-objective設計，權重是可學的，用MORL框架來解這問題，我們可以選擇不同λ平衡 reward 跟 constraint violation。必要時也可以針對特定場域進行λ tuning。

第二位報告人：張研究員（14:05-14:10）

張研究員：
我這邊補充ARAM在實作層面的設計。它採用兩個buffer：一個是合法transition的replay buffer，另一個是違規後loop的transition。這種結構非常像我們之前在做constraint-aware imitation learning時遇到的記憶共享問題。ARAM還加入了snapshot buffer，也就是MOSAC的Q Pensieve結構，讓不同行為偏好之間能共享學習記憶，這樣可顯著提升sample efficiency。

我認為這個架構對我們要解的多約束（如關節扭力＋視覺遮擋區域避障）問題，是很有幫助的。尤其是這種架構完全不需要我們設計新的projection operator，可以直接拿現成SAC加上accept-reject包裝，效能也比FlowPG或NFWPO來得穩定。

提問一：劉工程師
那張哥，我想問，這套方法有提到怎麼選accept-reject時的target distribution嗎？會不會選得不好導致學不到東西？

張研究員：
好問題。原文中有提一種策略是直接設π† ∝ π原始 restricted to feasible set，並且使用Student-t分布來增加探索性。實驗顯示這種選法對early stage exploration特別有幫助，而且不敏感於具體參數設定。

提問二：黃助理工程師
那這樣的multi-objective實作，會不會讓訓練時間大增？

張研究員：
他們的結果顯示即使是MORL架構，ARAM的training time比所有需要QP的方法都少兩到五個數量級。主因是QP次數幾乎是零，而且並行性高。

第三位報告人：劉工程師（14:10-14:15）

劉工程師：
我這邊聚焦在實驗結果上面，特別是可行性比例和推論效率。論文裡面提到，ARAM在所有MuJoCo與網路資源分配任務中都有最高的valid action rate，甚至在Ant環境達到100%。這代表這套方法不只訓練效率高，實際部署時也最穩。

另外他們針對inference latency測試100萬次推論，每次行為選擇平均花費44~63毫秒，比FlowPG或DPre+都快很多。而我們現在AGV推論速度目標是在50ms內，這是能達到的。更重要的是，ARAM方法完全移除掉前處理QP這件事，會讓我們整體部署架構更乾淨、維護性更高。

提問一：張研究員
那你覺得我們現有的路徑規劃模組要導入ARAM最需要調整的是哪一塊？

劉工程師：
主要是reward設計跟constraint check要能獨立模組化，這樣我們才能讓accept-reject模組做動作篩選。Policy結構可以保留我們原來的。

提問二：李協理
如果要做demo，大概要多久能初步整合進我們的開源控制平台？

劉工程師：
我保守估計兩週可以弄一版可跑demo的。因為架構簡單，難的是驗證我們實際的constraint設計是不是容易被學習到，這需要一點調參。

總結（14:15-14:20）

李協理：
感謝三位報告，也感謝大家的問題。結論是這套ARAM方法適合我們行為受限控制需求，特別是低計算負擔與部署簡潔性這兩點非常關鍵。請張工、劉工先開一個技術導入專案，7月中前給我一個demo評估結果。會議到此結束。
"""
with open("meeting.txt", "w", encoding="utf-8") as f:
    f.write(meeting_text)


# from google.colab import userdata

# print(userdata.get("OPENAI_API_KEY"))

import os, re, json, asyncio
from datetime import datetime, timedelta
from textwrap import dedent
# from openai import OpenAI
from kuwa.client import KuwaClient

MODEL = "gpt-4o-mini"
START_DATE = "2025-09-01"
TITLE = "會議任務 → 甘特圖"

def read_minutes_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def build_prompt_from_minutes(minutes_text: str) -> str:
    return dedent(f"""
    你是一位專案排程助理。請閱讀以下會議記錄，抽取目標(goals)、任務(tasks)、工期(duration_days)與依賴(depends_on)，並標示關鍵路徑。
    僅輸出**單一 JSON**，不要額外文字或 Markdown。

    會議記錄：
    \"\"\"{minutes_text.strip()}\"\"\"

    範例輸出結構：
    {{
      "goals": [{{"id":"G1","name":"本週重點"}}],
      "tasks": [
        {{"id":"T1","name":"API 串接","goal_id":"G1","duration_days":3,"depends_on":[]}},
        {{"id":"T2","name":"CI/CD 配置","goal_id":"G1","duration_days":2,"depends_on":["T1"]}}
      ],
      "critical_path": ["T1","T2"]
    }}
    """)

async def kuwa_shim(generator):
    result = ""
    async for chunk in generator:
        result += result
    return result

def call_openai(prompt: str) -> str:
    # client = OpenAI(
    #     api_key=userdata.get("OPENAI_API_KEY"),
    #     base_url="http://127.0.0.1/v1.0/"
    # )
    client = KuwaClient(
        base_url="http://127.0.0.1",
        model=".bot/Llama 3.2 3B @NPU",
        auth_token="7d63e6046ca2262d79ae41a8749d960e0ab86677a318086e69e5bb770fe6ec52",
    )
    resp = client.chat_complete(
        messages=[{"role": "system", "content": "只輸出 JSON"},
                  {"role": "user", "content": prompt}],
        # max_tokens=1000,
        # temperature=0.2
    )

    result = asyncio.run(kuwa_shim(resp))
    print(result)
    
    return result

def extract_json(text: str) -> dict:
    from json_repair import repair
    fixed_json = repair(text)
    m = re.search(r"(\{.*\})", fixed_json, flags=re.S)
    print(m)
    if not m:
        raise ValueError("沒找到 JSON")
    return json.loads(m.group(1))

def schedule(tasks: list) -> dict:
    t_by_id = {t["id"]: t for t in tasks}
    starts, ends = {}, {}

    def dfs(tid):
        if tid in ends:
            return ends[tid]
        deps = t_by_id[tid].get("depends_on", [])
        start = 0
        for d in deps:
            start = max(start, dfs(d))
        dur = int(max(1, t_by_id[tid].get("duration_days", 1)))
        starts[tid] = start
        ends[tid] = start + dur
        return ends[tid]

    for tid in t_by_id:
        dfs(tid)
    return {"start": starts, "end": ends}

def to_mermaid(goals, tasks, sched, title, start_date):
    base = datetime.strptime(start_date, "%Y-%m-%d")
    goal_name = {g["id"]: g["name"] for g in goals}
    lines = ["gantt", "dateFormat  YYYY-MM-DD", f"title {title}"]
    for g in goals:
        lines.append(f"section {g['name']}")
        for t in tasks:
            if t["goal_id"] != g["id"]:
                continue
            s = sched["start"][t["id"]]
            dur = t["duration_days"]
            start_date_iso = (base + timedelta(days=s)).strftime("%Y-%m-%d")
            lines.append(f"{t['name']} :{t['id']}, {start_date_iso}, {dur}d")
    return "\n".join(lines)

def convert_mermaid_to_csv_with_chatgpt(mermaid_text: str, model: str = "gpt-4o-mini"):
    # api_key = userdata.get("OPENAI_API_KEY")
    # if not api_key:
    #     raise RuntimeError("未找到 OPENAI_API_KEY。請先在環境變數設定。")

    # client = OpenAI(api_key=api_key)
    # client = OpenAI(
    #     api_key=userdata.get("OPENAI_API_KEY"),
    #     base_url="http://127.0.0.1/v1.0/"
    # )
    client = KuwaClient(
        base_url="http://127.0.0.1",
        model=".bot/Llama 3.2 3B @NPU",
        auth_token="7d63e6046ca2262d79ae41a8749d960e0ab86677a318086e69e5bb770fe6ec52",
    )
    user_prompt = USER_PROMPT_TEMPLATE.format(mermaid=mermaid_text)

    # 使用 Responses API
    # resp = client.responses.create(
        # model=model,
    resp = client.chat_complete(
        message=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        # temperature=0.0,
    )

    # 取出文字輸出
    # content = resp.output_text if hasattr(resp, "output_text") else resp.choices[0].message.content
    content = asyncio.run(kuwa_shim(resp))
    # 去掉前後空白
    return content.strip()


SYSTEM_PROMPT = """你是一個格式轉換器。把 Mermaid Gantt 語法轉換成 CSV（UTF-8, 逗號分隔）。
只輸出 CSV 內容，不要任何多餘解釋或前後綴。
CSV 欄位固定為：名稱,開始時間,結束時間,專案分類
規則：
- 從 `section <名稱>` 取得「專案分類」，直到下一個 section 前的任務都屬於該分類。
- 任務行格式類似：`任務名稱 :ID, YYYY-MM-DD, Nd` 或 `任務名稱 :YYYY-MM-DD, Nd`。
- 日期格式輸出為 YYYY-MM-DD（不含時間）。
- 若給的是開始日 + 持續天數 Nd，結束日 = 開始日 + (N-1) 天（含起訖）。
- 任務名稱輸出到「名稱」欄。
- 沒有 section 時，「專案分類」留空字串。
- 只產生資料列，不要加 CSV 標頭以外的任何文字。
"""

USER_PROMPT_TEMPLATE = """以下是 Mermaid Gantt 文字，請依規則轉成 CSV：
{mermaid}
請以這個 CSV 標頭開頭（務必包含、且順序固定）：
名稱,開始時間,結束時間,專案分類
"""

# === 主流程 ===
minutes = read_minutes_txt("meeting.txt")
prompt = build_prompt_from_minutes(minutes)
raw = call_openai(prompt)
data = extract_json(raw)

sched = schedule(data["tasks"])
mermaid = to_mermaid(data["goals"], data["tasks"], sched, TITLE, START_DATE)

print("=== Mermaid 甘特圖 ===")
print(mermaid)

csv = convert_mermaid_to_csv_with_chatgpt(mermaid)
print(csv)


def save_csv(csv_text: str, output_path: str = "notion_gantt.csv"):
    # 確保有換行結尾，避免某些工具讀取最後一列時被忽略
    if not csv_text.endswith("\n"):
        csv_text += "\n"
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        f.write(csv_text)
    print(f"✅ 已輸出 CSV：{os.path.abspath(output_path)}")
save_csv(csv)