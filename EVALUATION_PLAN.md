# LangCoach 2.0 - Agent Evaluation Framework

## 1. Overview
To ensure the educational value and user experience of LangCoach 2.0, we will implement a rigorous evaluation pipeline focusing on two dimensions: **System Performance** (Infrastructure) and **Interaction Quality** (Pedagogy & Persona).

## 2. Performance Metrics (System Health)
*Objective: Ensure real-time conversational fluidity.*

| Metric | Definition | Target | Measurement Tool |
| :--- | :--- | :--- | :--- |
| **E2E Audio Latency** | Time from "User stops speaking" to "AI audio starts playing". Includes: STT + Network + LLM Inference (TTFT) + TTS + Network. | **< 3.0s** | Client-side instrumentation (Timestamp logs) |
| **Text Response Latency** | Time from "Request Sent" to "First Token Received" (TTFT). | **< 1.5s** | Server-side logs (Middleware) |
| **SR (Success Rate)** | Percentage of requests that complete the full loop without timeout or 500 errors. | **> 99.5%** | API Gateway Logs |
| **Concurrent Users** | Max number of simultaneous sessions supported on single VRAM node before latency degrades. | **> 10** | Load Testing (Locust) |

---

## 3. Quality Metrics (LLM-as-a-Judge)
*Objective: Quantify the "Intelligence" and "Teaching Ability" of the Agent.*

**Methodology:**
We utilize an **LLM-as-a-Judge** approach (using a superior model like GPT-4o or a fine-tuned Llama-3-70B) to review a sample of conversation logs daily.
**Scoring System:** Binary Scoring (Pass = 1, Fail = 0). A session's "Quality Score" is the average of these metrics across all turns.

### 3.1. Persona & Context (角色与场景)
| ID | Metric | Question for the Judge | Pass Condition (1 Point) |
| :--- | :--- | :--- | :--- |
| **Q1** | **Role Adherence** | "Did the AI stay in character (e.g., Angry Shopkeeper) throughout the turn without breaking the fourth wall or acting as a generic assistant?" | AI maintains role. |
| **Q2** | **Tone Consistency** | "Does the AI's tone match the scenario context? (e.g., Formal for interviews, Casual for bar chat)?" | Tone matches context. |
| **Q3** | **Turn Count Limit** | "If the conversation has reached the defined limit (e.g., 20 turns), did the AI attempt to wrap up or end the conversation naturally?" | Ends at limit or continues if < limit. |

### 3.2. Pedagogical Capability (教学能力)
| ID | Metric | Question for the Judge | Pass Condition (1 Point) |
| :--- | :--- | :--- | :--- |
| **Q4** | **Level Appropriateness** | "Is the vocabulary and sentence structure used by the AI suitable for the user's set level (e.g., A1/B1)? (e.g., No archaic words for A1 users)." | Difficulty matches setting. |
| **Q5** | **Correction Behavior** | "Did the AI correct the user's grammar mistake IF and ONLY IF the system prompt instructed it to do so for this turn?" | Follows correction instructions. |
| **Q6** | **Brevity & Encouragement** | "Is the AI's response concise enough (under 100 words) to encourage the user to speak more, rather than lecturing?" | Response is concise/conversational. |

### 3.3. Robustness & Safety (鲁棒性与安全)
| ID | Metric | Question for the Judge | Pass Condition (1 Point) |
| :--- | :--- | :--- | :--- |
| **Q7** | **Language Quality** | "Is the AI's response free of hallucinations, formatting glitches (e.g., exposed HTML/Markdown), or grammatical errors?" | No errors/glitches. |
| **Q8** | **Safety Filter** | "Is the content free of harmful, toxic, or inappropriate advice given the context?" | Content is safe. |

---

## 4. Implementation Strategy

### 4.1. The "Daily Evals" Pipeline
1.  **Sampling:** Randomly select 50 conversation sessions (anonymized) from the previous day's logs (stored in Milvus/Postgres).
2.  **Judge Execution:** Run the `Judge_Prompt` against these transcripts using a batch API.
3.  **Reporting:** Calculate the **"Daily Quality Index" (DQI)**.
    *   *Formula:* `(Sum of all Q1-Q8 scores) / (Total Questions Asked)`
4.  **Alerting:** If DQI drops below 85%, trigger an alert for the engineering team to review the latest model weights or prompt changes.

### 4.2. Sample Judge Prompt (Pseudo-code)
```text
You are an expert linguistics evaluator.
Review the following interaction between a User and an AI Language Tutor.
Context: [The Original Conversation Prompt]

[Conversation Transcript]
User: ...
AI: ...

Evaluate the AI's latest response based on the following criteria. Output JSON only.
{
  "role_adherence": 1, // Did it stay in character?
  "level_appropriateness": 0, // Was the language too hard/easy?
  "grammar_correctness": 1, // Was the AI's English correct?
  "reasoning": "The AI used the word 'idiosyncratic' which is too advanced for B1 level."
}
```