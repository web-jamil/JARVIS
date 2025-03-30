import json
import os
import re
import getpass
from datetime import datetime
import csv
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import webbrowser

class Contact:
    def __init__(self, name, phone, email, country_code="+1", job_title=None, 
                 salary=None, department=None, company=None, notes=None):
        self.name = name
        self.phone = phone
        self.email = email
        self.country_code = country_code
        self.job_title = job_title
        self.salary = float(salary) if salary else None
        self.department = department
        self.company = company
        self.notes = notes
        self.created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.last_updated = self.created_at

    def __str__(self):
        info = [
            f"Name: {self.name}",
            f"Phone: {self.country_code}{self.phone}",
            f"Email: {self.email}",
            f"Company: {self.company}" if self.company else None,
            f"Position: {self.job_title}" if self.job_title else None,
            f"Department: {self.department}" if self.department else None,
            f"Salary: ${self.salary:,.2f}" if self.salary else None,
            f"Notes: {self.notes}" if self.notes else None,
            f"Last Updated: {self.last_updated}"
        ]
        return ", ".join(filter(None, info))

    def to_dict(self):
        return {
            "name": self.name,
            "phone": self.phone,
            "email": self.email,
            "country_code": self.country_code,
            "job_title": self.job_title,
            "salary": self.salary,
            "department": self.department,
            "company": self.company,
            "notes": self.notes,
            "created_at": self.created_at,
            "last_updated": self.last_updated
        }

    @staticmethod
    def from_dict(data):
        return Contact(
            data["name"],
            data["phone"],
            data["email"],
            data.get("country_code", "+1"),
            data.get("job_title"),
            data.get("salary"),
            data.get("department"),
            data.get("company"),
            data.get("notes")
        )

class ContactBook:
    def __init__(self, filename="contacts.json"):
        self.filename = filename
        self.contacts = self.load_contacts()
        self.country_codes = self.load_country_codes()
        self.departments = [
            "Engineering", "Marketing", "Sales", 
            "HR", "Finance", "Operations", "Other"
        ]
        self.history = []
        self.load_default_job_titles()

    def load_default_job_titles(self):
        self.job_titles = {
            "Engineering": ["Software Engineer", "DevOps", "QA Engineer", "CTO"],
            "Marketing": ["CMO", "Digital Marketer", "Content Writer"],
            "Sales": ["Sales Manager", "Account Executive", "Business Development"],
            "HR": ["HR Manager", "Recruiter", "Talent Acquisition"],
            "Finance": ["CFO", "Accountant", "Financial Analyst"],
            "Operations": ["COO", "Operations Manager", "Logistics"]
        }

    def load_country_codes(self):
        # Can be loaded from external JSON file
        return {
            "US": "+1", "UK": "+44", "IN": "+91",
            "AU": "+61", "DE": "+49", "FR": "+33",
            "JP": "+81", "BR": "+55", "CN": "+86"
        }

    def load_contacts(self):
        if not os.path.exists(self.filename):
            return {}
            
        try:
            with open(self.filename, 'r') as file:
                data = json.load(file)
                return {contact["name"]: Contact.from_dict(contact) for contact in data}
        except Exception as e:
            print(f"Error loading contacts: {e}")
            return {}

    def save_contacts(self):
        try:
            with open(self.filename, 'w') as file:
                json.dump([contact.to_dict() for contact in self.contacts.values()], 
                         file, indent=2)
        except Exception as e:
            print(f"Error saving contacts: {e}")

    def add_contact(self, contact):
        if contact.name in self.contacts:
            return False, "Contact already exists"
            
        self.contacts[contact.name] = contact
        self.save_contacts()
        self.log_action(f"Added contact: {contact.name}")
        return True, "Contact added successfully"

    def update_contact(self, name, **kwargs):
        if name not in self.contacts:
            return False, "Contact not found"
            
        contact = self.contacts[name]
        changes = []
        
        for key, value in kwargs.items():
            if value is not None and getattr(contact, key) != value:
                old_value = getattr(contact, key)
                setattr(contact, key, value)
                changes.append(f"{key} from {old_value} to {value}")
        
        if changes:
            contact.last_updated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.save_contacts()
            change_log = ", ".join(changes)
            self.log_action(f"Updated {name}: {change_log}")
            return True, "Contact updated successfully"
        return False, "No changes made"

    def delete_contact(self, name):
        if name not in self.contacts:
            return False, "Contact not found"
            
        del self.contacts[name]
        self.save_contacts()
        self.log_action(f"Deleted contact: {name}")
        return True, "Contact deleted successfully"

    def search_contacts(self, query, field="all"):
        results = []
        query = query.lower()
        
        for contact in self.contacts.values():
            if field == "all":
                fields_to_search = [
                    contact.name.lower(),
                    contact.phone,
                    contact.email.lower(),
                    str(contact.job_title).lower() if contact.job_title else "",
                    str(contact.company).lower() if contact.company else "",
                    str(contact.department).lower() if contact.department else ""
                ]
                if any(query in field for field in fields_to_search):
                    results.append(contact)
            else:
                field_value = getattr(contact, field, "").lower()
                if query in field_value:
                    results.append(contact)
                    
        return results

    def advanced_search(self, **criteria):
        results = []
        for contact in self.contacts.values():
            match = True
            for field, value in criteria.items():
                if value and getattr(contact, field, "").lower() != value.lower():
                    match = False
                    break
            if match:
                results.append(contact)
        return results

    def export_to_csv(self, filename):
        try:
            with open(filename, 'w', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=[
                    "name", "phone", "email", "country_code", 
                    "company", "job_title", "department", 
                    "salary", "notes", "created_at", "last_updated"
                ])
                writer.writeheader()
                for contact in self.contacts.values():
                    writer.writerow(contact.to_dict())
            self.log_action(f"Exported contacts to {filename}")
            return True, "Export successful"
        except Exception as e:
            return False, f"Export failed: {e}"

    def import_from_csv(self, filename):
        try:
            with open(filename, 'r') as file:
                reader = csv.DictReader(file)
                imported = 0
                for row in reader:
                    contact = Contact.from_dict(row)
                    self.contacts[contact.name] = contact
                    imported += 1
                self.save_contacts()
                self.log_action(f"Imported {imported} contacts from {filename}")
                return True, f"Imported {imported} contacts"
        except Exception as e:
            return False, f"Import failed: {e}"

    def get_stats(self):
        stats = {
            "total_contacts": len(self.contacts),
            "contacts_with_salary": sum(1 for c in self.contacts.values() if c.salary),
            "departments": defaultdict(int),
            "companies": defaultdict(int),
            "salary_stats": {
                "total": 0,
                "average": 0,
                "max": 0,
                "min": float('inf')
            }
        }
        
        total_salary = 0
        salary_count = 0
        
        for contact in self.contacts.values():
            if contact.department:
                stats["departments"][contact.department] += 1
            if contact.company:
                stats["companies"][contact.company] += 1
            if contact.salary:
                salary_count += 1
                total_salary += contact.salary
                stats["salary_stats"]["max"] = max(stats["salary_stats"]["max"], contact.salary)
                stats["salary_stats"]["min"] = min(stats["salary_stats"]["min"], contact.salary)
        
        if salary_count > 0:
            stats["salary_stats"]["total"] = total_salary
            stats["salary_stats"]["average"] = total_salary / salary_count
            stats["salary_stats"]["count"] = salary_count
        else:
            stats["salary_stats"]["min"] = 0
            
        return stats

    def generate_reports(self):
        stats = self.get_stats()
        reports = []
        
        # Department Distribution
        if stats["departments"]:
            dept_report = "\nDepartment Distribution:\n"
            for dept, count in sorted(stats["departments"].items(), key=lambda x: x[1], reverse=True):
                dept_report += f"{dept}: {count} contacts\n"
            reports.append(dept_report)
        
        # Company Distribution
        if stats["companies"]:
            company_report = "\nCompany Distribution:\n"
            for company, count in sorted(stats["companies"].items(), key=lambda x: x[1], reverse=True):
                company_report += f"{company}: {count} contacts\n"
            reports.append(company_report)
        
        # Salary Analysis
        if stats["salary_stats"]["count"] > 0:
            salary_report = "\nSalary Analysis:\n"
            salary_report += f"Total Salary: ${stats['salary_stats']['total']:,.2f}\n"
            salary_report += f"Average Salary: ${stats['salary_stats']['average']:,.2f}\n"
            salary_report += f"Highest Salary: ${stats['salary_stats']['max']:,.2f}\n"
            salary_report += f"Lowest Salary: ${stats['salary_stats']['min']:,.2f}\n"
            reports.append(salary_report)
        
        return "\n".join(reports)

    def plot_salary_distribution(self):
        salaries = [c.salary for c in self.contacts.values() if c.salary]
        if not salaries:
            return None
            
        plt.figure(figsize=(10, 6))
        plt.hist(salaries, bins=10, edgecolor='black', alpha=0.7)
        plt.title("Salary Distribution")
        plt.xlabel("Salary ($)")
        plt.ylabel("Number of Contacts")
        plt.grid(True)
        return plt

    def log_action(self, action):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.history.append(f"{timestamp} - {action}")
        if len(self.history) > 100:  # Keep last 100 actions
            self.history.pop(0)

    def get_activity_log(self):
        return "\n".join(self.history[-10:])  # Show last 10 actions

class ContactBookGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Professional Contact Management System")
        self.root.geometry("1200x800")
        
        self.contact_book = ContactBook()
        self.current_sort = "name"
        self.current_view = "all"
        
        self.setup_ui()
        self.load_contacts()
        
    def setup_ui(self):
        # Configure styles
        style = ttk.Style()
        style.configure("Treeview", rowheight=25)
        style.configure("TButton", padding=6)
        
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel - Search/Filter
        left_panel = ttk.Frame(main_frame, width=250)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # Search section
        search_frame = ttk.LabelFrame(left_panel, text="Search Contacts", padding=10)
        search_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.search_var = tk.StringVar()
        ttk.Entry(search_frame, textvariable=self.search_var).pack(fill=tk.X)
        ttk.Button(search_frame, text="Search", command=self.search_contacts).pack(fill=tk.X, pady=5)
        
        # Filter section
        filter_frame = ttk.LabelFrame(left_panel, text="Filter By", padding=10)
        filter_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(filter_frame, text="Department:").pack(anchor=tk.W)
        self.dept_filter = ttk.Combobox(filter_frame, values=self.contact_book.departments)
        self.dept_filter.pack(fill=tk.X)
        
        ttk.Label(filter_frame, text="Company:").pack(anchor=tk.W, pady=(5, 0))
        self.company_filter = ttk.Entry(filter_frame)
        self.company_filter.pack(fill=tk.X)
        
        ttk.Button(filter_frame, text="Apply Filters", command=self.apply_filters).pack(fill=tk.X, pady=5)
        ttk.Button(filter_frame, text="Clear Filters", command=self.clear_filters).pack(fill=tk.X)
        
        # Right panel - Main content
        right_panel = ttk.Frame(main_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Toolbar
        toolbar = ttk.Frame(right_panel)
        toolbar.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(toolbar, text="Add Contact", command=self.add_contact_dialog).pack(side=tk.LEFT)
        ttk.Button(toolbar, text="Edit Selected", command=self.edit_contact_dialog).pack(side=tk.LEFT, padx=5)
        ttk.Button(toolbar, text="Delete Selected", command=self.delete_contact).pack(side=tk.LEFT)
        
        ttk.Button(toolbar, text="Export CSV", command=self.export_csv).pack(side=tk.RIGHT)
        ttk.Button(toolbar, text="Import CSV", command=self.import_csv).pack(side=tk.RIGHT, padx=5)
        ttk.Button(toolbar, text="Reports", command=self.show_reports).pack(side=tk.RIGHT)
        
        # Contacts table
        self.tree = ttk.Treeview(right_panel, columns=("name", "company", "position", "phone", "email"), 
                               selectmode="extended")
        self.tree.heading("#0", text="ID")
        self.tree.heading("name", text="Name", command=lambda: self.sort_by("name"))
        self.tree.heading("company", text="Company", command=lambda: self.sort_by("company"))
        self.tree.heading("position", text="Position", command=lambda: self.sort_by("job_title"))
        self.tree.heading("phone", text="Phone", command=lambda: self.sort_by("phone"))
        self.tree.heading("email", text="Email", command=lambda: self.sort_by("email"))
        
        self.tree.column("#0", width=50, stretch=tk.NO)
        self.tree.column("name", width=150)
        self.tree.column("company", width=150)
        self.tree.column("position", width=150)
        self.tree.column("phone", width=120)
        self.tree.column("email", width=200)
        
        vsb = ttk.Scrollbar(right_panel, orient="vertical", command=self.tree.yview)
        hsb = ttk.Scrollbar(right_panel, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        
        self.tree.pack(fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        hsb.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Status bar
        self.status = ttk.Label(right_panel, text=f"Loaded {len(self.contact_book.contacts)} contacts", 
                              relief=tk.SUNKEN)
        self.status.pack(fill=tk.X, pady=(10, 0))
        
        # Configure tags for alternating colors
        self.tree.tag_configure('oddrow', background='#f0f0f0')
        self.tree.tag_configure('evenrow', background='#ffffff')
        
        # Double-click to edit
        self.tree.bind("<Double-1>", self.edit_contact_dialog)
        
    def load_contacts(self, contacts=None):
        self.tree.delete(*self.tree.get_children())
        
        contacts = contacts if contacts else self.contact_book.contacts.values()
        contacts = sorted(contacts, key=lambda x: getattr(x, self.current_sort).lower())
        
        for i, contact in enumerate(contacts):
            tags = ('evenrow',) if i % 2 == 0 else ('oddrow',)
            self.tree.insert("", tk.END, iid=i, text=str(i+1),
                            values=(contact.name, contact.company, contact.job_title, 
                                   f"{contact.country_code}{contact.phone}", contact.email),
                            tags=tags)
        
        self.status.config(text=f"Showing {len(contacts)} contacts")
        
    def search_contacts(self):
        query = self.search_var.get().strip()
        if not query:
            self.load_contacts()
            return
            
        results = self.contact_book.search_contacts(query)
        self.load_contacts(results)
        
    def apply_filters(self):
        filters = {}
        dept = self.dept_filter.get().strip()
        company = self.company_filter.get().strip()
        
        if dept:
            filters["department"] = dept
        if company:
            filters["company"] = company
            
        if filters:
            results = self.contact_book.advanced_search(**filters)
            self.load_contacts(results)
            self.current_view = "filtered"
        else:
            self.clear_filters()
            
    def clear_filters(self):
        self.dept_filter.set("")
        self.company_filter.delete(0, tk.END)
        self.load_contacts()
        self.current_view = "all"
        
    def sort_by(self, column):
        self.current_sort = {
            "name": "name",
            "company": "company",
            "position": "job_title",
            "phone": "phone",
            "email": "email"
        }[column]
        
        self.load_contacts()
        
    def add_contact_dialog(self):
        dialog = ContactDialog(self.root, self.contact_book, self)
        self.root.wait_window(dialog.top)
        
    def edit_contact_dialog(self, event=None):
        selected = self.tree.selection()
        if not selected:
            messagebox.showwarning("No Selection", "Please select a contact to edit")
            return
            
        contact_id = selected[0]
        contact_values = self.tree.item(contact_id, "values")
        contact_name = contact_values[0]
        
        if contact_name in self.contact_book.contacts:
            contact = self.contact_book.contacts[contact_name]
            dialog = ContactDialog(self.root, self.contact_book, self, contact)
            self.root.wait_window(dialog.top)
        
    def delete_contact(self):
        selected = self.tree.selection()
        if not selected:
            messagebox.showwarning("No Selection", "Please select a contact to delete")
            return
            
        contact_names = [self.tree.item(iid, "values")[0] for iid in selected]
        
        if messagebox.askyesno("Confirm Delete", f"Delete {len(contact_names)} selected contacts?"):
            for name in contact_names:
                if name in self.contact_book.contacts:
                    self.contact_book.delete_contact(name)
            self.load_contacts()
            
    def export_csv(self):
        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")],
            title="Save Contacts As"
        )
        if filename:
            success, message = self.contact_book.export_to_csv(filename)
            if success:
                messagebox.showinfo("Success", message)
            else:
                messagebox.showerror("Error", message)
                
    def import_csv(self):
        filename = filedialog.askopenfilename(
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")],
            title="Select Contacts File"
        )
        if filename:
            success, message = self.contact_book.import_from_csv(filename)
            if success:
                self.load_contacts()
                messagebox.showinfo("Success", message)
            else:
                messagebox.showerror("Error", message)
                
    def show_reports(self):
        report_window = tk.Toplevel(self.root)
        report_window.title("Contact Reports")
        report_window.geometry("800x600")
        
        notebook = ttk.Notebook(report_window)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        # Text Report Tab
        text_frame = ttk.Frame(notebook)
        notebook.add(text_frame, text="Summary")
        
        report_text = tk.Text(text_frame, wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(text_frame, command=report_text.yview)
        report_text.config(yscrollcommand=scrollbar.set)
        
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        report_text.pack(fill=tk.BOTH, expand=True)
        
        report_text.insert(tk.END, self.contact_book.generate_reports())
        report_text.config(state=tk.DISABLED)
        
        # Salary Distribution Tab
        salary_frame = ttk.Frame(notebook)
        notebook.add(salary_frame, text="Salary Distribution")
        
        fig = self.contact_book.plot_salary_distribution()
        if fig:
            canvas = FigureCanvasTkAgg(fig, master=salary_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        else:
            ttk.Label(salary_frame, text="No salary data available").pack(pady=20)
        
        # Activity Log Tab
        log_frame = ttk.Frame(notebook)
        notebook.add(log_frame, text="Activity Log")
        
        log_text = tk.Text(log_frame, wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(log_frame, command=log_text.yview)
        log_text.config(yscrollcommand=scrollbar.set)
        
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        log_text.pack(fill=tk.BOTH, expand=True)
        
        log_text.insert(tk.END, self.contact_book.get_activity_log())
        log_text.config(state=tk.DISABLED)

class ContactDialog:
    def __init__(self, parent, contact_book, gui, contact=None):
        self.top = tk.Toplevel(parent)
        self.contact_book = contact_book
        self.gui = gui
        self.contact = contact
        self.is_edit = contact is not None
        
        self.top.title("Edit Contact" if self.is_edit else "Add New Contact")
        self.top.geometry("500x600")
        
        self.create_widgets()
        if self.is_edit:
            self.load_contact_data()
        
    def create_widgets(self):
        main_frame = ttk.Frame(self.top, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Form fields
        ttk.Label(main_frame, text="Name*:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.name_var = tk.StringVar()
        ttk.Entry(main_frame, textvariable=self.name_var).grid(row=0, column=1, sticky=tk.EW, padx=5, pady=5)
        
        ttk.Label(main_frame, text="Country Code:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.country_var = tk.StringVar(value="+1")
        country_frame = ttk.Frame(main_frame)
        country_frame.grid(row=1, column=1, sticky=tk.EW, padx=5, pady=5)
        ttk.Entry(country_frame, textvariable=self.country_var, width=5).pack(side=tk.LEFT)
        ttk.Button(country_frame, text="Select", command=self.select_country).pack(side=tk.LEFT, padx=5)
        
        ttk.Label(main_frame, text="Phone*:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.phone_var = tk.StringVar()
        ttk.Entry(main_frame, textvariable=self.phone_var).grid(row=2, column=1, sticky=tk.EW, padx=5, pady=5)
        
        ttk.Label(main_frame, text="Email*:").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.email_var = tk.StringVar()
        ttk.Entry(main_frame, textvariable=self.email_var).grid(row=3, column=1, sticky=tk.EW, padx=5, pady=5)
        
        ttk.Label(main_frame, text="Company:").grid(row=4, column=0, sticky=tk.W, pady=5)
        self.company_var = tk.StringVar()
        ttk.Entry(main_frame, textvariable=self.company_var).grid(row=4, column=1, sticky=tk.EW, padx=5, pady=5)
        
        ttk.Label(main_frame, text="Department:").grid(row=5, column=0, sticky=tk.W, pady=5)
        self.dept_var = tk.StringVar()
        dept_combo = ttk.Combobox(main_frame, textvariable=self.dept_var, 
                                 values=self.contact_book.departments)
        dept_combo.grid(row=5, column=1, sticky=tk.EW, padx=5, pady=5)
        dept_combo.bind("<<ComboboxSelected>>", self.update_job_titles)
        
        ttk.Label(main_frame, text="Job Title:").grid(row=6, column=0, sticky=tk.W, pady=5)
        self.job_var = tk.StringVar()
        self.job_combo = ttk.Combobox(main_frame, textvariable=self.job_var)
        self.job_combo.grid(row=6, column=1, sticky=tk.EW, padx=5, pady=5)
        
        ttk.Label(main_frame, text="Salary:").grid(row=7, column=0, sticky=tk.W, pady=5)
        self.salary_var = tk.StringVar()
        ttk.Entry(main_frame, textvariable=self.salary_var).grid(row=7, column=1, sticky=tk.EW, padx=5, pady=5)
        
        ttk.Label(main_frame, text="Notes:").grid(row=8, column=0, sticky=tk.W, pady=5)
        self.notes_var = tk.StringVar()
        ttk.Entry(main_frame, textvariable=self.notes_var).grid(row=8, column=1, sticky=tk.EW, padx=5, pady=5)
        
        # Button frame
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=9, column=0, columnspan=2, pady=20)
        
        ttk.Button(button_frame, text="Save", command=self.save_contact).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=self.top.destroy).pack(side=tk.LEFT, padx=5)
        
        # Configure grid weights
        main_frame.columnconfigure(1, weight=1)
        
    def select_country(self):
        country_window = tk.Toplevel(self.top)
        country_window.title("Select Country Code")
        
        for code, prefix in self.contact_book.country_codes.items():
            ttk.Button(
                country_window, 
                text=f"{code}: {prefix}", 
                command=lambda p=prefix: self.set_country_and_close(p, country_window)
            ).pack(fill=tk.X, padx=5, pady=2)
            
        ttk.Label(country_window, text="Or enter custom:").pack(pady=(10, 0))
        custom_var = tk.StringVar()
        ttk.Entry(country_window, textvariable=custom_var).pack(pady=5)
        ttk.Button(
            country_window, 
                text="Use Custom", 
                command=lambda: self.set_country_and_close(custom_var.get(), country_window)
        ).pack(pady=5)
        
    def set_country_and_close(self, prefix, window):
        if re.match(r'^\+\d{1,3}$', prefix):
            self.country_var.set(prefix)
            window.destroy()
        else:
            messagebox.showerror("Invalid", "Please enter a valid country code (e.g., +33)")
        
    def update_job_titles(self, event=None):
        dept = self.dept_var.get()
        if dept in self.contact_book.job_titles:
            self.job_combo["values"] = self.contact_book.job_titles[dept]
        else:
            self.job_combo["values"] = []
        
    def load_contact_data(self):
        self.name_var.set(self.contact.name)
        self.country_var.set(self.contact.country_code)
        self.phone_var.set(self.contact.phone)
        self.email_var.set(self.contact.email)
        self.company_var.set(self.contact.company if self.contact.company else "")
        self.dept_var.set(self.contact.department if self.contact.department else "")
        self.job_var.set(self.contact.job_title if self.contact.job_title else "")
        self.salary_var.set(str(self.contact.salary) if self.contact.salary else "")
        self.notes_var.set(self.contact.notes if self.contact.notes else "")
        
        if self.contact.department:
            self.update_job_titles()
        
    def validate_fields(self):
        errors = []
        
        if not self.name_var.get().strip():
            errors.append("Name is required")
            
        if not self.phone_var.get().strip():
            errors.append("Phone number is required")
        elif not re.match(r'^\d{10,15}$', self.phone_var.get()):
            errors.append("Phone must be 10-15 digits")
            
        if not self.email_var.get().strip():
            errors.append("Email is required")
        elif not re.match(r'^[\w\.-]+@[\w\.-]+\.\w+$', self.email_var.get()):
            errors.append("Invalid email format")
            
        if self.salary_var.get().strip():
            try:
                float(self.salary_var.get())
            except ValueError:
                errors.append("Salary must be a number")
                
        if errors:
            messagebox.showerror("Validation Error", "\n".join(errors))
            return False
        return True
        
    def save_contact(self):
        if not self.validate_fields():
            return
            
        contact_data = {
            "name": self.name_var.get().strip(),
            "phone": self.phone_var.get().strip(),
            "email": self.email_var.get().strip(),
            "country_code": self.country_var.get().strip(),
            "company": self.company_var.get().strip() or None,
            "department": self.dept_var.get().strip() or None,
            "job_title": self.job_var.get().strip() or None,
            "salary": float(self.salary_var.get()) if self.salary_var.get().strip() else None,
            "notes": self.notes_var.get().strip() or None
        }
        
        if self.is_edit:
            success, message = self.contact_book.update_contact(self.contact.name, **contact_data)
        else:
            new_contact = Contact(**contact_data)
            success, message = self.contact_book.add_contact(new_contact)
            
        if success:
            self.gui.load_contacts()
            self.top.destroy()
        else:
            messagebox.showerror("Error", message)

if __name__ == "__main__":
    root = tk.Tk()
    app = ContactBookGUI(root)
    root.mainloop()