#  RAG-Based SEO Content Pipeline

Production RAG system built with n8n that generates 
SEO-optimized articles from herbal reference books 
stored on local NAS infrastructure.

## Workflow

<img width="784" height="286" alt="RAG workflow" src="https://github.com/user-attachments/assets/02b70117-46ac-41d6-bd86-6985f46a9047" />

##  Overview

Deployed at **Herbs Are My World** (Cyprus) as part of 
Erasmus+ AI Automation internship.

Combines:
-  RAG search on local NAS
-  Multi-LLM (Groq + Mistral)
-  Custom Python API
-  SEO optimization
-  Airtable + Telegram distribution

##  Pipeline Flow
Keyword Input
↓
Search Books (RAG on NAS)
↓
Clean Data
↓
Keyword Analyzer (Groq)
↓
Intent Detector (Mistral)
↓
SEO Outline Generator (Mistral)
↓
Python Article Generator
↓
Clean & Humanize
↓
Airtable + Telegram


## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Automation | n8n (self-hosted) |
| RAG API | Custom Python |
| Storage | NAS (local) |
| LLMs | Groq, Mistral Cloud |
| Database | Airtable |
| Distribution | Telegram |

##  Results

| Metric | Value |
|--------|-------|
| Google SEO Score | 5.8/10 |
| AI Detection | 33% |
| Automation | End-to-end |
| Data | Private NAS |

##  Author
**Sarath Kumar Radhakrishnan**
-  M.Sc. Cybersecurity | Riga Technical University
-  Erasmus+ Intern | Herbs Are My World, Cyprus
- sarathkumar.rrk@gmail.com
-  [LinkedIn](https://www.linkedin.com/in/sarath-kumar-radhakrishnan/)

## License
MIT License
