import io
import shutil
import sqlite3
import sys
import tkinter as tk
from tkinter import messagebox, filedialog
from tkinter import ttk
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class MachineLearningApp:
    def __init__(self):
        self.data_ax = None
        self.data_fig = None
        self.window = tk.Tk()
        self.window.title("Machine Learning App - Kacper Kicior")

        self.data = None
        self.model = None

        self.button_frame = tk.Frame(self.window)
        self.button_frame.pack()

        self.load_data_button = tk.Button(self.button_frame, text="Load Data", command=self.load_data)
        self.load_data_button.pack(side=tk.LEFT, padx=10, pady=10)

        self.train_button = tk.Button(self.button_frame, text="Train Model", command=self.train_model,
                                      state=tk.DISABLED)
        self.train_button.pack(side=tk.LEFT, padx=10, pady=10)

        self.test_button = tk.Button(self.button_frame, text="Test Model", command=self.test_model, state=tk.DISABLED)
        self.test_button.pack(side=tk.LEFT, padx=10, pady=10)

        self.predict_button = tk.Button(self.button_frame, text="Predict", command=self.predict, state=tk.DISABLED)
        self.predict_button.pack(side=tk.LEFT, padx=10, pady=10)

        self.add_data_button = tk.Button(self.button_frame, text="Add Data", command=self.add_data, state=tk.DISABLED)
        self.add_data_button.pack(side=tk.LEFT, padx=10, pady=10)

        self.rebuild_button = tk.Button(self.button_frame, text="Rebuild Model", command=self.rebuild_model,
                                        state=tk.DISABLED)
        self.rebuild_button.pack(side=tk.LEFT, padx=10, pady=10)

        self.visualize_button = tk.Button(self.button_frame, text="Visualize Data", command=self.visualize_data_table,
                                          state=tk.DISABLED)
        self.visualize_button.pack(side=tk.LEFT, padx=10, pady=10)

        self.data_table_frame = tk.Frame(self.window)
        self.data_table_frame.pack()

        self.data_table = ttk.Treeview(self.data_table_frame)
        self.data_table.pack(side=tk.LEFT)

        self.data_table_scrollbar = ttk.Scrollbar(self.data_table_frame, orient="vertical",
                                                  command=self.data_table.yview)
        self.data_table_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.data_table.configure(yscrollcommand=self.data_table_scrollbar.set)

        self.data_table["columns"] = ()
        self.data_table.heading("#0", text="Data")

        self.info_label = tk.Label(self.window, text="", fg="blue")
        self.info_label.pack(padx=10, pady=10)

        self.save_model_button = tk.Button(self.window, text="Save Model", command=self.save_database(),
                                           state=tk.DISABLED)
        self.save_model_button.pack(padx=10, pady=10)

        self.load_model_button = tk.Button(self.window, text="Load Model", command=self.load_database())
        self.load_model_button.pack(padx=10, pady=10)

    def load_data(self):
        file_path = filedialog.askopenfilename(
            filetypes=(("Data files", "*.data"), ("CSV files", "*.csv"), ("All files", "*.*")))
        if file_path:
            self.data = pd.read_csv(file_path)
            self.train_button.config(state=tk.NORMAL)
            self.test_button.config(state=tk.NORMAL)
            self.predict_button.config(state=tk.NORMAL)
            self.add_data_button.config(state=tk.NORMAL)
            self.rebuild_button.config(state=tk.NORMAL)
            self.visualize_button.config(state=tk.NORMAL)
            self.show_data_table()

    def show_data_table(self):
        self.data_table.delete(*self.data_table.get_children())
        columns = self.data.columns.tolist()
        self.data_table["columns"] = tuple(columns)
        self.data_table.heading("#0", text="Index")
        for column in columns:
            self.data_table.column(column, width=5)
            self.data_table.heading(column, text=column)
        for index, row in self.data.iterrows():
            self.data_table.insert("", "end", text=index, values=tuple(row))

    def visualize_data_table(self):
        pass
        # if self.data_fig is not None:
        #     self.data_fig.clear()
        #
        # selected_items = self.data_table
        # if not selected_items:
        #     self.info_label.config(text="No data selected.", fg="red")
        #     return
        #
        # selected_rows = [self.data_table.item(item)["text"] for item in selected_items]
        #
        # if len(selected_rows) > 1:
        #     self.info_label.config(text="Multiple data selection is not supported for visualization.", fg="red")
        #     return
        #
        # index = int(selected_rows[0])
        # row_values = self.data_table.item(selected_items[0])["values"]
        #
        # if self.data_fig is None:
        #     self.data_fig = plt.figure(figsize=(8, 5))
        #     self.data_ax = self.data_fig.add_subplot(111)
        #
        # self.data_ax.clear()
        # self.data_ax.bar(range(len(row_values)), row_values)
        # self.data_ax.set_xticks(range(len(row_values)))
        # self.data_ax.set_xticklabels(self.data.columns.tolist(), rotation=45)
        # self.data_ax.set_xlabel("Columns")
        # self.data_ax.set_ylabel("Values")
        # self.data_ax.set_title(f"Data Visualization for Index: {index}")
        #
        # plt.tight_layout()
        # plt.show(block=False)

    def train_model(self):
        X = self.data.iloc[:, :-1]
        y = self.data.iloc[:, -1]
        X_encoded = pd.get_dummies(X)
        X_train, _, y_train, _ = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
        self.model = RandomForestClassifier()
        self.model.fit(X_train, y_train)
        self.info_label.config(text="Model trained successfully.")
        self.save_model_button.config(state=tk.NORMAL)

    def test_model(self):
        X = self.data.iloc[:, :-1]
        y = self.data.iloc[:, -1]
        X_encoded = pd.get_dummies(X)
        _, X_test, _, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        self.info_label.config(text=f"Model tested successfully. Accuracy: {accuracy}")

    def predict(self):
        X = self.data.iloc[:, :-1]
        y = self.data.iloc[:, -1]
        X_encoded = pd.get_dummies(X)
        _, X_test, _, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
        prediction = self.model.predict(X_test.iloc[0].values.reshape(1, -1))
        self.info_label.config(text=f"New data prediction: {prediction}")

    def add_data(self):
        column_names = self.data.columns.tolist()
        new_data_values = []

        for column_name in column_names:
            new_value = self.prompt_user_input(column_name)
            new_data_values.append(new_value)

        new_data = pd.DataFrame([new_data_values], columns=column_names)
        self.data = pd.concat([self.data, new_data], ignore_index=True)

        self.show_data_table()
        self.info_label.config(text="New data added successfully.")

    def prompt_user_input(self, column_name):
        value = tk.simpledialog.askstring("Enter Value", f"Enter value for {column_name}")
        if value is None:
            self.info_label.config(text="New data addition cancelled.")
            self.window.focus_force()
            self.window.grab_set()
            self.window.wait_window()
        return value

    def rebuild_model(self):
        if self.data is not None:
            X = self.data.iloc[:, :-1]
            y = self.data.iloc[:, -1]
            X_encoded = pd.get_dummies(X)
            X_train, _, y_train, _ = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
            self.model = RandomForestClassifier()
            self.model.fit(X_train, y_train)
            self.info_label.config(text="Model rebuilt successfully.")
            self.save_model_button.config(state=tk.NORMAL)
        else:
            self.info_label.config(text="No data available. Load data before rebuilding the model.")

    def save_database(self):
        pass

    def load_database(self):
        pass

    def run(self):
        self.window.mainloop()


if __name__ == "__main__":
    app = MachineLearningApp()
    app.run()
