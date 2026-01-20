# Product Plan: LangCoach 2.0 - Advanced AI Language Learning Platform

## 1. Executive Summary
LangCoach 2.0 aims to evolve from a simple text-based scenario practice tool into a comprehensive, multi-modal, and personalized component-based language learning assistant. The platform will leverage advanced open-source local LLMs, specialized speech models, and persistent memory systems to simulate realistic conversations. The architecture allows for privacy-first, on-premise deployment using optimized inference engines (Ollama/vLLM) while extending accessibility through a WeChat Mini Program.

---

## 2. Core Feature Enhancements & Requirements

### 2.1. Conversation Customization
*   **Turn Control:** Users can select the length of the conversation to fit their study time window.
    *   *Options:* Short (10), Standard (20 - Default), Extended (30), Deep Dive (50).
    *   *Implementation:* Dynamic prompt injection into LangChain templates.
*   **Difficulty Scaling:** Adjusts the vocabulary range, grammar complexity, and speaking speed of the AI agent.
    *   *Levels:* Primary (A1/A2), Medium (B1/B2), Advanced (C1/C2).
    *   *Implementation:* Context variable passed to the System Prompt.

### 2.2. Multi-Modal Interaction (Speech)
*   **Deployment:** Fully private, locally hosted.
*   **Speech-to-Text (STT):**
    *   *Model:* OpenAI Whisper-v3 (Fine-tuned via Unsloth).
    *   *Goal:* Enhanced recognition of accented learner speech.
*   **Text-to-Speech (TTS):**
    *   *Model:* Orpheus (or similar high-quality openweights model, Fine-tuned).
    *   *Goal:* Natural prosody and emotive speech matching the scenario context.

### 2.3. AI & Infrastructure Stack
*   **Core Chat/Reasoning Model:** GLM-4-9B (4-bit quantization).
    *   *Reasoning:* Excellent Chinese-English bilingual ability and instruction following.
*   **Training/Fine-tuning:**
    *   *Tooling:* Unsloth (for efficient LoRA/QLoRA).
    *   *Data Sources:* Hugging Face datasets (e.g., Common Voice, LibriSpeech for audio; specialized ESL dialogue datasets for chat).
*   **Inference Engine:**
    *   *Dev/Local:* Ollama (ease of use).
    *   *Production:* vLLM (high throughput, continuous batching).
*   **Application Logic:** LangChain (orchestration).

### 2.4. Memory Architecture
*   **Short-term (Context):** LangChain `ConversationBufferWindowMemory` (managing the immediate input/output sliding window).
*   **Long-term (RAG/Profile):** Milvus Vector Database.
    *   *Storage:* User vocabulary mastery, past mistakes, preferred topics, and tone analysis.
    *   *Retrieval:* Re-injecting past corrections into new conversations to reinforce learning.

### 2.5. User Interface (Web)
*   **Framework:** Gradio (re-designed for "Fancy & Handy" UX).
*   **Components:**
    *   Chat Bubble Interface with Avatar support.
    *   Audio Recorder (Push-to-talk) & Audio Player components.
    *   Sidebar for configuration (Level, Turns, Scenario).
    *   Real-time feedback panel (grammar corrections).

### 2.6. User Interface (Mobile - WeChat Mini Program)
*   **Architecture:** Frontend (Mini Program) + Backend API (FastAPI wrapper around Gradio/LangChain).
*   **Key Features:** Audio-first interface, daily check-ins, social sharing of conversation scores.

### 2.7. Quality Assurance & Observability
*   **Evaluation Framework:**
    *   We follow a strict **"Performance + Quality"** dual-track evaluation strategy.
    *   Detailed metrics, pass/fail criteria (LLM-as-a-Judge), and CI/CD test cases are defined in the standalone document: **[EVALUATION_PLAN.md](./EVALUATION_PLAN.md)**.
    *   *Key Focus:* E2E Latency, strict prompt format compliance, and pedagogical logic verification.
*   **User Feedback (RLHF Data Collection):**
    *   *Granularity:* Per-turn thumbs up/down and Post-session 5-star rating.
    *   *Usage:* "Polly" mechanism to accept user signals for offline review and dataset augmentation.

---

## 3. Additional Recommended Features (Competitor Benchmarking)

To compete with apps like Duolingo, HelloTalk, or Talkpal AI, the following features are integrated into the plan:

1.  **AI Post-Session Analysis (The "Coach" Agent):**
    *   After the conversation ends, the AI generates a report card: *Grammar Score*, *Vocabulary Richness*, and *Actionable Improvement Tips*.
2.  **Scenario Randomizer:**
    *   "Surprise Me" button that generates dynamic scenarios (e.g., "Argue a parking ticket," "Explain a gap in your resume") using a meta-prompt.
3.  **Click-to-Translate & Dictionary:**
    *   In the text history, clicking a word shows the definition and adds it to the user's personal "Flashcard Deck" (stored in Milvus).
4.  **Role Reversal:**
    *   A mode where the *user* acts as the interviewer/clerk, and the *AI* acts as the applicant/customer, testing comprehension rather than production.

---

## 4. Development Phases & Roadmap

### Phase 1: Foundation & Customization (Weeks 1-3)
*   **Goal:** Enhance the text-based core.
*   **Tasks:**
    *   Refactor `prompts/` to use Jinja2 templates for variable injection (Turns, Levels).
    *   Update `ConversationAgent` in `src/agents/` to accept configuration parameters.
    *   Update Gradio UI sidebar to include Sliders/Dropdowns for these new settings.
    *   **Milestone:** v1.1.0 - Configurable Chat.

### Phase 2: The Infrastructure Migration (Weeks 4-6)
*   **Goal:** Switch to GLM-4 and set up Vector Memory.
*   **Tasks:**
    *   Set up local Docker environment for Milvus.
    *   Integrate `Unsloth` + `Ollama` workflow for GLM-4-9B-4bit.
    *   Implement Long-term memory logic: Create embeddings for conversation summaries and store in Milvus.
    *   Reference Check: Ensure retrieving memory context doesn't exceed context window limits.
    *   **Milestone:** v1.2.0 - Smart Memory & New LLM.

### Phase 3: The Voice Revolution (Weeks 7-11)
*   **Goal:** Add Ears and Mouth.
*   **Tasks:**
    *   **Dataset Prep:** Curate datasets from Hugging Face for STT/TTS fine-tuning.
    *   **Fine-tuning:** Run Unsloth jobs for Whisper-v3 and Orpheus.
    *   **Backend:** Create async API endpoints for `/transcribe` and `/synthesize`.
    *   **Frontend:** Add Microphone input and Audio output elements to Gradio.
    *   **Optimization:** Ensure latency is under 2 seconds for a conversational feel.
    *   **Milestone:** v2.0.0 - Voice functionality.

### Phase 4: Production & Mobile Expansion (Weeks 12-16)
*   **Goal:** Scale, Mobile access, and Observability.
*   **Tasks:**
    *   **Server:** Migrate inference to vLLM for concurrency.
    *   **API Layer:** Build a robust FastAPI middleware to expose Gradio functions to the outside world.
    *   **Analytics:** Implement middleware to log E2E latency per request. Store user ratings (Stars) in database.
    *   **WeChat Mini Program:**
        *   *Design:* UX mockups for mobile flow (including "Rate this Session" screens).
        *   *Dev:* Build pages (Login, Chat, Settings, Profile).
        *   *Auth:* Integrate WeChat Login.
        *   *Audio:* Handle WeChat recorder manager format conversion to server format (silk/mp3 -> wav).
    *   **Milestone:** v2.1.0 - Mobile Launch.

### Phase 5: Quality Assurance (Weeks 17-18)
*   **Goal:** Validate product quality against the defined evaluation framework.
*   **Tasks:**
    *   **Evaluation Implementation:**
        *   Execute the protocols defined in **[EVALUATION_PLAN.md](./EVALUATION_PLAN.md)**, covering E2E Latency targets, LLM-as-a-Judge scoring rules, and automated test cases.
    *   **Compliance Verification:**
        *   Ensure strict adherence to system prompts (e.g., format compliance, bilingual logic) and persona consistency across all scenarios.
*   **Milestone:** v2.2.0 - Quality Guard.

---

## 5. WeChat Mini Program Architecture Design

Since the heavy lifting (LLM/STT/TTS) happens on the private server, the Mini Program is a lightweight client.

### Architecture
1.  **Client (WeChat):** WXML/WXSS/JS. handling UI, Audio recording/playback, Network requests.
2.  **API Gateway:** Nginx handling SSL and routing.
3.  **Middleware (Python FastAPI):**
    *   Authenticates WeChat Users (User ID -> Milvus Collection ID).
    *   Queues requests to the Inference Server.
    *   Manages Session State (Redis).
4.  **Inference Layer:** The existing vLLM/Ollama setup.

### Screens
1.  **Home Tab:** "Daily Challenge," "Quick Practice" button, specialized "Topic Cards" (Business, Travel, Social).
2.  **Chat Tab:** The active conversation interface. Bubble UI. Hold-to-speak button (like WeChat voice messages) is the most intuitive interaction for this platform.
3.  **Review Tab:** Historical feedback reports generated by the AI. Flashcards based on saved words.
4.  **Profile Tab:** Statistics (Total minutes spoken, Level progress).

---

## 6. Iteration Plan

| Version | Release Name | Key Deliverables |
| :--- | :--- | :--- |
| **v1.1** | *The Config Update* | Variable prompts (Turns/Levels), modified LangChain logic. |
| **v1.2** | *The Brain Update* | Switch to GLM-4, Milvus Memory integration. |
| **v2.0** | *The Voice Update* | Local TTS/STT integration, Gradio Audio UI. |
| **v2.1** | *The Mobile Bridge* | API Layer exposure, vLLM migration. |
| **v3.0** | *WeChat Companion* | Full Mini Program release coupled with the backend. |
