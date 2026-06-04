# AstraZeneca



\# Local Data Science \& RAG Pipeline Engine



An isolated, high-performance data processing and local Retrieval-Augmented Generation (RAG) backend framework. This environment is strictly locked to \*\*Python 3.11.9\*\* to guarantee exact runtime architecture across all execution machines.



\---



\## System Architecture \& Prerequisites



Before setting up the project, ensure you have the following prerequisites installed on your host system:

1\. \*\*Python 3.11.9\*\* (Must be the exact micro-version patch).

2\. \*\*Ollama\*\* (With the `llama3` model pulled locally via `ollama run llama3`).



\---



\##  Local Installation \& Setup



Follow these exact steps in your terminal to safely stand up the workspace without breaking dependency trees or tracking massive file matrices on GitHub.



\### 1. Provision an Empty Sandbox Environment

Force your local machine to build an isolated virtual bubble using your specific Python 3.11 base interpreter:

```powershell

C:\\python3119\\python.exe -m venv .venv
.venv\\Scripts\\activate
pip install --upgrade pip

pip install -r requirements.txt


To create a jupyter notebook kernel please run the below mentioned command

python -m ipykernel install --user --name=astra-zeneca-env --display-name "Python 3.11 (AstraZeneca Engine)"

