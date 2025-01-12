import streamlit as st
import pandas as pd
from st_aggrid import AgGrid, GridOptionsBuilder
import io
import os

# Custom CSS
st.markdown("""
    <style>
        /* Main container */
        .main {
            background-color: #f5f5f5;
            padding: 20px;
        }
        /* Title styles */
        .title {
            font-size: 40px;
            color: #4CAF50;
            font-weight: bold;
            text-align: center;
            margin-bottom: 20px;
        }
        /* Description styles */
        .description {
            font-size: 18px;
            color: #555555;
            text-align: center;
            margin-bottom: 30px;
        }
        /* Button styles */
        .stButton button {
            font-size: 16px;
            background-color: #4CAF50;
            color: white;
            padding: 10px 20px;
            border-radius: 5px;
            border: none;
        }
        .stButton button:hover {
            background-color: #45a049;
        }
        /* Table container styles */
        .table-container {
            background-color: white;
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 15px;
            margin-bottom: 20px;
        }
        /* Uploaded file message */
        .stSuccess {
            background-color: #e6ffea;
            color: #333;
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 20px;
        }
    </style>
""", unsafe_allow_html=True)

def parse_file(file):
    """Parses the uploaded file and converts it into a DataFrame."""
    data = []
    file_content = file.getvalue().decode("utf-8").splitlines()
    for line in file_content:
        if line.startswith("D01"):
            data.append(line.split("|"))
    num_columns = len(data[0]) if data else 0
    columns = [f"Kolom{i+1}" for i in range(num_columns)]
    return pd.DataFrame(data, columns=columns), file_content

def save_to_txt(df, original_file_content, original_filename):
    """Saves the DataFrame back into a text file in the same format as the original."""
    output = io.StringIO()
    first_line = original_file_content[0].strip()
    header = first_line.split("|")
    output.write("|".join(header) + "\n")
    for row in df.values:
        output.write("|".join(map(str, row)) + "\n")
    content = output.getvalue()
    output.close()
    base_name = os.path.splitext(original_filename)[0]
    new_filename = f"{base_name}_diedit.txt"
    return content, new_filename

def main():
    st.markdown("<div class='main'>", unsafe_allow_html=True)
    st.markdown("<div class='title'>Konverter File ke Tabel dengan Edit</div>", unsafe_allow_html=True)
    st.markdown("<div class='description'>Unggah file teks untuk dikonversi menjadi tabel. Setelah itu, Anda dapat mengedit dan menyimpannya kembali.</div>", unsafe_allow_html=True)
    
    # Sidebar for file uploader
    st.sidebar.title("Unggah File")
    uploaded_file = st.sidebar.file_uploader("Pilih file teks", type=["txt"])
    
    if uploaded_file is not None:
        st.success("File berhasil diunggah!")
        df, original_file_content = parse_file(uploaded_file)
        
        # Display editable table
        st.markdown("<div class='table-container'>", unsafe_allow_html=True)
        grid_options = GridOptionsBuilder.from_dataframe(df)
        grid_options.configure_default_column(editable=True)
        grid_options = grid_options.build()
        edited_df = AgGrid(df, gridOptions=grid_options, enable_enterprise_modules=True, update_mode="MODEL_CHANGED")
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Display edited data
        st.write("## Tabel yang Telah Diedit")
        st.dataframe(edited_df['data'])
        
        # Buttons to save edited data
        st.write("## Simpan Tabel yang Telah Diedit")
        
        # Use columns for buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Simpan CSV"):
                edited_data = edited_df['data']
                csv = edited_data.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Unduh CSV",
                    data=csv,
                    file_name="tabel_konversi_diedit.csv",
                    mime="text/csv"
                )
        with col2:
            if st.button("Simpan TXT"):
                content, new_filename = save_to_txt(edited_df['data'], original_file_content, uploaded_file.name)
                st.download_button(
                    label="Unduh TXT",
                    data=content,
                    file_name=new_filename,
                    mime="text/plain"
                )
    
    st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()