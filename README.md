# PCA_Raman
Single-molecule verification using PCA

# Reference

https://www.nature.com/articles/s41467-018-07869-5

# run

## merge_data.py

**Overview**
A lightweight command-line tool that merges multiple `saved_col_n` files from a specified directory into a single `merged.csv`.

**Key Features**

* **Column Merging:** Retains the shared first column and horizontally appends the second column from each matching file.
* **Dynamic Headers:** Automatically renames the top row of each appended column using its source filename (without the extension).
* **Format Priority:** Prioritizes CSV files over TXT files if both exist in the directory.
* **Smart Sorting:** Sorts input files numerically by `n` (e.g., 1, 2, 10) rather than alphabetically.
* **CLI Execution:** Uses a single command-line argument as both the input directory and output destination.

run:

python merge_data.py "C:\Users\YourName\Desktop\DataFolder"



