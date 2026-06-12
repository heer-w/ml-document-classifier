import os
import random
from faker import Faker

fake = Faker()
random.seed(42)


BASE_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')

# ── 1. INVOICE ────────────────────────────────────────────────
def make_invoice():
    lines = [
        f"INVOICE",
        f"Invoice No: INV-{random.randint(1000,9999)}",
        f"Date: {fake.date()}",
        f"Bill To: {fake.company()}",
        f"Address: {fake.address()}",
        f"",
        f"Item          Qty    Unit Price    Total",
        f"{fake.word().capitalize()} Service   {random.randint(1,10)}    ${random.randint(50,500)}    ${random.randint(100,5000)}",
        f"{fake.word().capitalize()} Product   {random.randint(1,5)}     ${random.randint(20,200)}    ${random.randint(50,1000)}",
        f"",
        f"Subtotal: ${random.randint(500,5000)}",
        f"Tax (18% GST): ${random.randint(50,500)}",
        f"Total Amount Due: ${random.randint(600,6000)}",
        f"",
        f"Payment Due Date: {fake.date()}",
        f"Bank Account: {fake.bban()}",
        f"Thank you for your business.",
    ]
    return "\n".join(lines)

# ── 2. PURCHASE ORDER ─────────────────────────────────────────
def make_purchase_order():
    lines = [
        f"PURCHASE ORDER",
        f"PO Number: PO-{random.randint(1000,9999)}",
        f"Date: {fake.date()}",
        f"Vendor: {fake.company()}",
        f"Ship To: {fake.address()}",
        f"",
        f"Ordered By: {fake.name()}",
        f"Department: {fake.bs().capitalize()}",
        f"",
        f"Item Description     Quantity    Unit Cost    Total Cost",
        f"{fake.word().capitalize()}            {random.randint(1,50)}          ${random.randint(10,200)}       ${random.randint(100,5000)}",
        f"{fake.word().capitalize()}            {random.randint(1,20)}          ${random.randint(5,100)}        ${random.randint(50,2000)}",
        f"",
        f"Total Order Value: ${random.randint(500,8000)}",
        f"Delivery Date: {fake.date()}",
        f"Terms: Net 30",
        f"Authorized Signature: {fake.name()}",
    ]
    return "\n".join(lines)

# ── 3. RESUME ─────────────────────────────────────────────────
def make_resume():
    skills = ["Python", "Machine Learning", "SQL", "Java", "Data Analysis",
              "TensorFlow", "React", "Node.js", "Excel", "Communication"]
    lines = [
        f"{fake.name()}",
        f"Email: {fake.email()}  |  Phone: {fake.phone_number()}",
        f"LinkedIn: linkedin.com/in/{fake.user_name()}",
        f"",
        f"OBJECTIVE",
        f"{fake.sentence(nb_words=20)}",
        f"",
        f"EDUCATION",
        f"B.Tech in {random.choice(['Computer Science','Electronics','Mechanical'])}",
        f"{fake.company()} University, {fake.year()}",
        f"GPA: {round(random.uniform(7.0, 9.9), 2)}",
        f"",
        f"EXPERIENCE",
        f"{fake.job()} at {fake.company()} ({fake.year()} - Present)",
        f"{fake.sentence(nb_words=15)}",
        f"",
        f"{fake.job()} at {fake.company()} ({fake.year()} - {fake.year()})",
        f"{fake.sentence(nb_words=15)}",
        f"",
        f"SKILLS",
        f"{', '.join(random.sample(skills, 6))}",
        f"",
        f"CERTIFICATIONS",
        f"{fake.bs().capitalize()} Certification - {fake.company()}",
    ]
    return "\n".join(lines)

# ── 4. BANK STATEMENT ─────────────────────────────────────────
def make_bank_statement():
    lines = [
        f"BANK STATEMENT",
        f"{fake.company()} Bank",
        f"Account Holder: {fake.name()}",
        f"Account Number: {fake.bban()}",
        f"Statement Period: {fake.date()} to {fake.date()}",
        f"",
        f"Date         Description                  Debit      Credit     Balance",
    ]
    balance = random.randint(10000, 50000)
    for _ in range(8):
        date = fake.date()
        desc = fake.company()
        amount = random.randint(100, 5000)
        if random.random() > 0.5:
            balance -= amount
            lines.append(f"{date}   {desc:<28} ${amount:<10}            ${balance}")
        else:
            balance += amount
            lines.append(f"{date}   {desc:<28}            ${amount:<10} ${balance}")
    lines += [
        f"",
        f"Closing Balance: ${balance}",
        f"Minimum Balance Required: $1000",
        f"Interest Earned: ${random.randint(10,200)}",
    ]
    return "\n".join(lines)

# ── 5. EMAIL ──────────────────────────────────────────────────
def make_email():
    lines = [
        f"From: {fake.email()}",
        f"To: {fake.email()}",
        f"CC: {fake.email()}",
        f"Date: {fake.date_time().strftime('%a, %d %b %Y %H:%M:%S')}",
        f"Subject: {fake.sentence(nb_words=6)}",
        f"",
        f"Dear {fake.first_name()},",
        f"",
        f"{fake.paragraph(nb_sentences=3)}",
        f"",
        f"{fake.paragraph(nb_sentences=3)}",
        f"",
        f"Please let me know if you have any questions.",
        f"",
        f"Best regards,",
        f"{fake.name()}",
        f"{fake.job()}",
        f"{fake.company()}",
        f"Phone: {fake.phone_number()}",
    ]
    return "\n".join(lines)

# ── GENERATE ALL 100 DOCS ─────────────────────────────────────
generators = {
    "Invoice":        make_invoice,
    "Purchase_Order": make_purchase_order,
    "Resume":         make_resume,
    "Bank_Statement": make_bank_statement,
    "Email":          make_email,
}

for class_name, func in generators.items():
    folder = os.path.join(BASE_DIR, class_name)
    for i in range(1, 21):  # 20 docs per class
        content = func()
        filename = f"doc_{i:03d}.txt"
        with open(os.path.join(folder, filename), "w") as f:
            f.write(content)
    print(f"✅ Generated 20 documents for: {class_name}")

print("\n🎉 Done! 100 documents created in data/raw/")