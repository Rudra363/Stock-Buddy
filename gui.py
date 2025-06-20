import tkinter as tk
import customtkinter
import sys
import os

class Registration:
    def __init__(self, master):
        customtkinter.set_appearance_mode("dark")
        customtkinter.set_default_color_theme("dark-blue")

        self.master = master
        self.master.title("Stock Buddy")
        self.master.geometry("900x600")

        self.frame = customtkinter.CTkFrame(master=self.master)
        self.frame.pack(pady=20, padx=60, fill="both", expand=True)

        self.label = customtkinter.CTkLabel(master=self.frame, text="Enter Stock", font=("Arial", 24))
        self.label.pack(pady=12, padx=10)

        self.entry1 = customtkinter.CTkEntry(master=self.frame, placeholder_text="Enter Stock", width=1)
        self.entry1.pack(pady=12, padx=10, expand=True, fill='both', anchor='center')

        self.checkbox = customtkinter.CTkCheckBox(master=self.frame, text="Already Have an Account? Login In", command=self.login)
        self.checkbox.pack(pady=12, padx=10)

        self.button = customtkinter.CTkButton(master=self.frame, text="Next", command=self.register)
        self.button.pack(pady=12, padx=10)

    def register(self):
        user_name = self.entry1.get()
        # Save to database or perform registration logic here
        print(f"Registered user: {user_name}")
       # create(user_name)

    def login(self):
        print("Placeholder for login functionality")
        self.everything()


#Create main window and run
# if __name__ == "__main__":
#     root = tk.Tk()
#     app = Registration(root)
#     root.mainloop()# import customtkinter