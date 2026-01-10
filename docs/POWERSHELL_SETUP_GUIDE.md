# 🎯 QVERIFY GitHub Repository Setup Guide
## Complete Step-by-Step PowerShell Commands

**Author**: H M Shujaat Zaheer  
**Email**: shujabis@gmail.com  
**GitHub**: https://github.com/hmshujaatzaheer

---

## 📋 TABLE OF CONTENTS

1. [Prerequisites](#1-prerequisites)
2. [Install Git](#2-install-git)
3. [Configure Git](#3-configure-git)
4. [Create GitHub Repository](#4-create-github-repository)
5. [Clone Repository](#5-clone-repository)
6. [Copy Project Files](#6-copy-project-files)
7. [Stage All Files](#7-stage-all-files)
8. [Commit Files](#8-commit-files)
9. [Push to GitHub](#9-push-to-github)
10. [Verify Upload](#10-verify-upload)
11. [Troubleshooting](#11-troubleshooting)

---

## 📖 REPOSITORY INFORMATION

| Property | Value |
|----------|-------|
| **Repository Name** | `qverify-framework` |
| **Repository URL** | `https://github.com/hmshujaatzaheer/qverify-framework` |
| **Description** | QVERIFY: LLM-Assisted Formal Verification of Quantum Programs - A framework integrating Large Language Model reasoning with formal verification of quantum programs |
| **Topics/Tags** | quantum-computing, formal-verification, llm, specification-synthesis, smt-solving, python, quantum-programs, silq, openqasm, machine-learning |
| **License** | MIT |
| **Visibility** | Public |

---

## 1️⃣ PREREQUISITES

### What You Need Before Starting:

1. ✅ Windows Computer (Windows 10 or 11)
2. ✅ Internet Connection
3. ✅ GitHub Account (https://github.com/hmshujaatzaheer)
4. ✅ The QVERIFY project files (downloaded from Claude)

### Open PowerShell:

```powershell
# Step 1: Press Windows Key + X
# Step 2: Click "Windows PowerShell" or "Terminal"
# OR
# Step 1: Press Windows Key
# Step 2: Type "PowerShell"
# Step 3: Press Enter
```

---

## 2️⃣ INSTALL GIT

### Check if Git is Already Installed:

```powershell
# Type this command and press Enter:
git --version
```

**If you see a version number** (like `git version 2.43.0`), skip to Step 3.

**If you see an error**, install Git:

### Install Git Using Winget (Windows 10/11):

```powershell
# Type this command and press Enter:
winget install --id Git.Git -e --source winget
```

### OR Download Git Manually:

1. Go to: https://git-scm.com/download/win
2. Download the installer
3. Run the installer
4. Click "Next" for all options (use defaults)
5. Click "Install"
6. Click "Finish"

### Restart PowerShell After Installing Git:

```powershell
# Close PowerShell (type this or click X):
exit

# Open PowerShell again (Windows Key → type PowerShell → Enter)
```

### Verify Git Installation:

```powershell
# Type this command and press Enter:
git --version

# You should see something like:
# git version 2.43.0.windows.1
```

---

## 3️⃣ CONFIGURE GIT

### Set Your Name (The Name That Appears on Commits):

```powershell
# Type this command (replace with your actual name):
git config --global user.name "H M Shujaat Zaheer"
```

### Set Your Email (Must Match Your GitHub Email):

```powershell
# Type this command (replace with your actual email):
git config --global user.email "shujabis@gmail.com"
```

### Set Default Branch Name:

```powershell
# Type this command:
git config --global init.defaultBranch main
```

### Verify Your Configuration:

```powershell
# Type this command:
git config --global --list

# You should see:
# user.name=H M Shujaat Zaheer
# user.email=shujabis@gmail.com
# init.defaultbranch=main
```

---

## 4️⃣ CREATE GITHUB REPOSITORY

### Option A: Create Repository on GitHub Website (RECOMMENDED)

1. **Open your web browser**
2. **Go to**: https://github.com/new
3. **Fill in the form**:

   | Field | Value |
   |-------|-------|
   | Repository name | `qverify-framework` |
   | Description | `QVERIFY: LLM-Assisted Formal Verification of Quantum Programs - A framework integrating Large Language Model reasoning with formal verification of quantum programs` |
   | Public/Private | ✅ Public |
   | Add README | ❌ Do NOT check (we have our own) |
   | Add .gitignore | ❌ Do NOT check (we have our own) |
   | Choose a license | ❌ None (we have our own) |

4. **Click**: "Create repository"

### Option B: Create Repository Using GitHub CLI

```powershell
# First, install GitHub CLI:
winget install --id GitHub.cli -e --source winget

# Login to GitHub:
gh auth login
# Follow the prompts (choose GitHub.com, HTTPS, and login via browser)

# Create the repository:
gh repo create qverify-framework --public --description "QVERIFY: LLM-Assisted Formal Verification of Quantum Programs"
```

---

## 5️⃣ CLONE REPOSITORY

### Navigate to Your Desired Folder:

```powershell
# Go to your Documents folder (or any folder you prefer):
cd $HOME\Documents

# OR go to a specific folder:
cd C:\Users\YourUsername\Projects
```

### Clone the Empty Repository:

```powershell
# Type this command:
git clone https://github.com/hmshujaatzaheer/qverify-framework.git

# You should see:
# Cloning into 'qverify-framework'...
# warning: You appear to have cloned an empty repository.
```

### Navigate Into the Repository:

```powershell
# Type this command:
cd qverify-framework

# Verify you're in the right folder:
pwd

# You should see something like:
# Path
# ----
# C:\Users\YourUsername\Documents\qverify-framework
```

---

## 6️⃣ COPY PROJECT FILES

### Option A: If You Downloaded Files from Claude as ZIP

```powershell
# First, extract the ZIP file to a temporary location
# Then copy all contents to the repository folder

# If your ZIP was extracted to Downloads\qverify-framework-files:
Copy-Item -Path "$HOME\Downloads\qverify-framework-files\*" -Destination "." -Recurse -Force
```

### Option B: If Files Are Already in a Folder

```powershell
# Replace 'C:\path\to\qverify-files' with your actual path:
Copy-Item -Path "C:\path\to\qverify-files\*" -Destination "." -Recurse -Force
```

### Verify Files Were Copied:

```powershell
# List all files and folders:
Get-ChildItem -Recurse | Select-Object FullName

# You should see files like:
# README.md
# pyproject.toml
# requirements.txt
# src\qverify\__init__.py
# ... and many more
```

### Quick Check for Key Files:

```powershell
# Check if main files exist:
Test-Path "README.md"
Test-Path "pyproject.toml"
Test-Path "src\qverify\__init__.py"
Test-Path "LICENSE"

# All should return: True
```

---

## 7️⃣ STAGE ALL FILES

### Add All Files to Git Staging:

```powershell
# Type this command:
git add .

# The dot (.) means "add everything in current folder"
```

### Verify Files Are Staged:

```powershell
# Type this command:
git status

# You should see a list of green "new file:" entries like:
# new file:   .env.example
# new file:   .gitignore
# new file:   CHANGELOG.md
# new file:   CONTRIBUTING.md
# new file:   LICENSE
# new file:   README.md
# new file:   pyproject.toml
# ... and many more
```

---

## 8️⃣ COMMIT FILES

### Create Your First Commit:

```powershell
# Type this command:
git commit -m "Initial commit: QVERIFY framework for LLM-assisted quantum program verification"
```

### What the Commit Message Means:
- `-m` = "message"
- The text in quotes describes what you did

### Verify Commit Was Created:

```powershell
# Type this command:
git log --oneline

# You should see something like:
# abc1234 (HEAD -> main) Initial commit: QVERIFY framework...
```

---

## 9️⃣ PUSH TO GITHUB

### Push Your Commits to GitHub:

```powershell
# Type this command:
git push -u origin main
```

### If Asked for Credentials:

**Option A: Browser Authentication (Recommended)**
- A browser window will open
- Click "Authorize" or login to GitHub
- Return to PowerShell

**Option B: Personal Access Token**
1. Go to: https://github.com/settings/tokens
2. Click "Generate new token (classic)"
3. Give it a name like "QVERIFY Push"
4. Check these permissions:
   - ✅ repo (all)
5. Click "Generate token"
6. Copy the token
7. When prompted for password in PowerShell, paste the token

### Verify Push Succeeded:

```powershell
# You should see output like:
# Enumerating objects: 123, done.
# Counting objects: 100% (123/123), done.
# ...
# To https://github.com/hmshujaatzaheer/qverify-framework.git
#  * [new branch]      main -> main
# branch 'main' set up to track 'origin/main'.
```

---

## 🔟 VERIFY UPLOAD

### Check on GitHub Website:

1. **Open your browser**
2. **Go to**: https://github.com/hmshujaatzaheer/qverify-framework
3. **You should see**:
   - All your files listed
   - README.md displayed at the bottom
   - Green "Latest commit" message

### Verify in PowerShell:

```powershell
# Check remote status:
git remote -v

# You should see:
# origin  https://github.com/hmshujaatzaheer/qverify-framework.git (fetch)
# origin  https://github.com/hmshujaatzaheer/qverify-framework.git (push)

# Check branch status:
git status

# You should see:
# On branch main
# Your branch is up to date with 'origin/main'.
# nothing to commit, working tree clean
```

---

## 🏷️ ADD REPOSITORY TOPICS (OPTIONAL BUT RECOMMENDED)

### On GitHub Website:

1. Go to: https://github.com/hmshujaatzaheer/qverify-framework
2. Click the ⚙️ gear icon next to "About" (right side)
3. In "Topics" field, add these (press Enter after each):
   - `quantum-computing`
   - `formal-verification`
   - `llm`
   - `specification-synthesis`
   - `smt-solving`
   - `python`
   - `quantum-programs`
   - `silq`
   - `openqasm`
   - `machine-learning`
   - `phd-research`
4. Click "Save changes"

---

## 📝 COMPLETE POWERSHELL COMMAND SUMMARY

### Copy-Paste Ready (Run These in Order):

```powershell
# ============================================
# QVERIFY GITHUB SETUP - ALL COMMANDS
# ============================================

# 1. Check Git version (must be installed)
git --version

# 2. Configure Git (run once, ever)
git config --global user.name "H M Shujaat Zaheer"
git config --global user.email "shujabis@gmail.com"
git config --global init.defaultBranch main

# 3. Navigate to your projects folder
cd $HOME\Documents

# 4. Clone the repository (create it on GitHub first!)
git clone https://github.com/hmshujaatzaheer/qverify-framework.git

# 5. Enter the repository
cd qverify-framework

# 6. Copy your project files here (adjust path as needed)
# Copy-Item -Path "C:\path\to\your\qverify-files\*" -Destination "." -Recurse -Force

# 7. Stage all files
git add .

# 8. Check what's staged
git status

# 9. Commit
git commit -m "Initial commit: QVERIFY framework for LLM-assisted quantum program verification"

# 10. Push to GitHub
git push -u origin main

# 11. Verify
git status
git log --oneline
```

---

## 🔧 TROUBLESHOOTING

### Problem: "git is not recognized"

```powershell
# Solution: Restart PowerShell after installing Git
# Or add Git to PATH manually:
$env:Path += ";C:\Program Files\Git\cmd"
```

### Problem: "Permission denied" when pushing

```powershell
# Solution: Use a Personal Access Token
# 1. Go to https://github.com/settings/tokens
# 2. Generate a new token with 'repo' permissions
# 3. Use the token as your password when prompted
```

### Problem: "Repository not found"

```powershell
# Solution: Make sure the repository exists on GitHub
# 1. Go to https://github.com/new
# 2. Create the repository first
# 3. Then try cloning again
```

### Problem: "Merge conflict" or "Updates were rejected"

```powershell
# Solution: Pull first, then push
git pull origin main --allow-unrelated-histories
git push origin main
```

### Problem: Large files won't push

```powershell
# Solution: Remove large files from tracking
git rm --cached large_file.zip
git commit -m "Remove large file"
git push origin main
```

---

## ✅ SUCCESS CHECKLIST

After completing all steps, verify:

- [ ] Repository visible at https://github.com/hmshujaatzaheer/qverify-framework
- [ ] README.md displays correctly with badges and structure
- [ ] All folders visible: `src/`, `tests/`, `docs/`, `configs/`, etc.
- [ ] LICENSE file present
- [ ] pyproject.toml present
- [ ] .github/workflows/ci.yml present (for CI/CD)

---

## 🎉 CONGRATULATIONS!

Your QVERIFY repository is now live on GitHub!

**Repository URL**: https://github.com/hmshujaatzaheer/qverify-framework

**Next Steps**:
1. Star your own repository ⭐
2. Share the link in your PhD proposal
3. Update the arXiv badge when you publish
4. Add collaborators if needed

---

*Guide created for QVERIFY PhD Proposal by H M Shujaat Zaheer*
