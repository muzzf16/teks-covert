import streamlit as st
import pandas as pd
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode
import io
import os
import re

# Set page config must be the first streamlit command
st.set_page_config(layout="wide", page_title="Konverter File D01")

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

def get_column_name(n):
    """Converts a 1-based number to Excel-style column name (A, B, ... Z, AA, AB...)."""
    result = ""
    while n > 0:
        n, remainder = divmod(n - 1, 26)
        result = chr(65 + remainder) + result
    return result

def parse_file(file):
    """Parses the uploaded file and converts it into a DataFrame, tracking D01 indices."""
    data = []
    d01_indices = []
    file_content = file.getvalue().decode("utf-8").splitlines()
    
    for i, line in enumerate(file_content):
        if line.startswith("D01"):
            data.append(line.split("|"))
            d01_indices.append(i)
            
    # Calculate max columns to handle jagged rows
    num_columns = max(len(row) for row in data) if data else 0
    columns = [get_column_name(i+1) for i in range(num_columns)]
    
    # Pad rows to ensure uniform length
    # This prevents "10 columns passed, passed data had 19 columns" errors if using DataFrame constructor directly with mismatch
    # Actually DataFrame(data) handles it but 'columns' arg must match width? 
    # If we pass explicit columns list, it expects data to match.
    # It is safer to DataFrame from list of lists and THEN set columns, OR pad manually.
    # DataFrame constructor with 'columns' will trunc or error if mismatch?
    # Let's simple create DataFrame and let it infer, then set columns? No, logic needs columns.
    
    df = pd.DataFrame(data)
    # If more columns than we thought? No, we used max.
    # If less columns? It fills None.
    
    # Rename columns 
    # df.columns might be 0..N-1.
    df.columns = columns[:len(df.columns)] 
    # Wait, if we use max, len(df.columns) should equal num_columns.
    
    return df, file_content, d01_indices

def save_to_txt(df, original_file_content, d01_indices, original_filename):
    """Saves the DataFrame back into a text file by replacing the D01 block."""
    if not d01_indices:
        # Fallback if no D01 indices were found originally (shouldn't happen if df is populated)
        return "\r\n".join(original_file_content), original_filename

    # Determine the start and end of the original D01 block
    # Assuming D01 lines are roughly contiguous or we just replace the range from min to max
    start_idx = min(d01_indices)
    end_idx = max(d01_indices)

    # 1. Get header (lines before first D01)
    header_lines = original_file_content[:start_idx]
    
    # 2. Get footer (lines after last D01)
    # Note: If there were non-D01 lines interleaved, they will be LOST with this logic.
    # Given the constraint of "adding/removing rows", preserving exact interleaved non-D01 lines is complex.
    # We will assume standard format where D01 is a block.
    footer_lines = original_file_content[end_idx+1:]
    
    # 3. Build new D01 block from DataFrame
    new_d01_lines = []
    # Drop _index if present
    save_df = df.drop(columns=['_index'], errors='ignore')
    
    for _, row in save_df.iterrows():
        # Clean data: Replace NaN/None with empty string
        row_clean = row.fillna("").astype(str).replace("nan", "").replace("None", "")
        # Join with pipes
        line_content = "|".join(row_clean.tolist())
        new_d01_lines.append(line_content)
        
    # Combine
    final_lines = header_lines + new_d01_lines + footer_lines
    content = "\r\n".join(final_lines)
    
    # Ensure trailing newline if it's standard text file
    content += "\r\n"
    
    base_name = os.path.splitext(original_filename)[0]
    new_filename = f"{base_name}_diedit.txt"
    return content, new_filename

def main():

    st.markdown("<div class='main'>", unsafe_allow_html=True)
    st.markdown("<div class='title'>Konverter File ke Tabel dengan Edit</div>", unsafe_allow_html=True)
    st.markdown("<div class='description'>Unggah satu atau lebih file teks untuk dikonversi menjadi tabel. Anda dapat mengedit setiap file di tab masing-masing dan menyimpannya kembali tanpa menghilangkan Header/Footer.</div>", unsafe_allow_html=True)
    
    # Sidebar for file uploader
    st.sidebar.title("Unggah File")
    uploaded_files = st.sidebar.file_uploader("Pilih file teks", type=["txt"], accept_multiple_files=True)
    
    if uploaded_files:
        st.success(f"{len(uploaded_files)} File berhasil diunggah!")
        
        # Create tabs for each file
        tab_names = [file.name for file in uploaded_files]
        tabs = st.tabs(tab_names)
        
        for i, (tab, file) in enumerate(zip(tabs, uploaded_files)):
            with tab:
                st.subheader(f"Editing: {file.name}")
                
                # Use session state to store data logic
                ss_key = f"data_{file.name}"
                
                if ss_key not in st.session_state:
                    df_parsed, original_file_content_parsed, d01_indices_parsed = parse_file(file)
                    st.session_state[ss_key] = {
                        "df": df_parsed,
                        "original_content": original_file_content_parsed,
                        "d01_indices": d01_indices_parsed
                    }
                
                # Load from session state
                data_state = st.session_state[ss_key]
                df = data_state["df"]
                original_file_content = data_state["original_content"]
                d01_indices = data_state["d01_indices"]
                
                if df.empty:
                    st.warning(f"Tidak ditemukan baris data yang dimulai dengan 'D01' di file {file.name}.")
                else:
                    # --- Toolbar: Edit Struktur (Baris & Kolom) ---
                    # Placed ABOVE AgGrid
                    # Search Bar
                    search_term = st.text_input("Cari Baris (Search)", placeholder="Ketik untuk memfilter data...", key=f"search_{file.name}")
                    
                    st.write("##### Edit Struktur Tabel")
                    tb_col1, tb_col2 = st.columns([1, 1])

                    grid_key = f"grid_{file.name}_{i}"
                    
                    # Access PREVIOUS selection from session state for "Buttons Above" logic
                    prev_selection = []
                    # Debugging Selection
                    # st.write(f"Debug Grid Key: {grid_key}")
                    if grid_key in st.session_state and st.session_state[grid_key] is not None:
                         prev_selection = st.session_state[grid_key].get("selectedItems", [])
                    
                    if not prev_selection:
                        # Fallback: Check if we can get it from the last run's variable if it exists in session? 
                        # No, simpler to just rely on the key.
                        pass

                    # Row Controls
                    with tb_col1:
                        # Use sub-columns for compact buttons "Row: [+] [-]"
                        r_c1, r_c2, r_c3 = st.columns([0.4, 0.3, 0.3])
                        with r_c1: st.write("**Baris:**")
                        with r_c2:
                            if st.button("➕", key=f"btn_add_row_{file.name}", help="Sisipkan Baris di Bawah Seleksi"):
                                current_df = st.session_state[ss_key]["df"]
                                new_row = {c: "" for c in current_df.columns if c != "_index"}
                                # Default value
                                if not current_df.empty:
                                    first_val = current_df.iloc[0, 0]
                                    if str(first_val).startswith("D01"):
                                        new_row[current_df.columns[0]] = "D01"
                                
                                # Use prev_selection
                                if prev_selection:
                                    sel_ids = [r.get("_index") for r in prev_selection]
                                    positions = current_df.index[current_df['_index'].isin(sel_ids)].tolist()
                                    if positions:
                                        insert_pos = max(positions) + 1
                                        part1 = current_df.iloc[:insert_pos]
                                        part2 = current_df.iloc[insert_pos:]
                                        new_df = pd.concat([part1, pd.DataFrame([new_row]), part2]).reset_index(drop=True)
                                    else:
                                        new_df = pd.concat([current_df, pd.DataFrame([new_row])], ignore_index=True)
                                else:
                                    new_df = pd.concat([current_df, pd.DataFrame([new_row])], ignore_index=True)

                                new_df["_index"] = range(len(new_df))
                                st.session_state[ss_key]["df"] = new_df
                                if hasattr(st, 'rerun'): st.rerun()
                                else: st.experimental_rerun()

                        with r_c3:
                            if st.button("➖", key=f"btn_del_row_{file.name}", help="Hapus Baris Terpilih"):
                                if not prev_selection:
                                    st.warning("Pilih baris di tabel dulu!")
                                else:
                                    current_df = st.session_state[ss_key]["df"]
                                    sel_ids = [r.get("_index") for r in prev_selection]
                                    new_df = current_df[~current_df['_index'].isin(sel_ids)].reset_index(drop=True)
                                    new_df["_index"] = range(len(new_df))
                                    st.session_state[ss_key]["df"] = new_df
                                    if hasattr(st, 'rerun'): st.rerun()
                                    else: st.experimental_rerun()

                    # Column Controls
                    with tb_col2:
                         c_c1, c_c2, c_c3, c_c4 = st.columns([0.3, 0.2, 0.2, 0.3])
                         with c_c1: st.write("**Kolom:**")
                         with c_c2:
                             if st.button("➕", key=f"btn_add_col_{file.name}", help="Tambah Kolom di Akhir"):
                                 current_df = st.session_state[ss_key]["df"]
                                 clean_cols = [c for c in current_df.columns if c not in ["_index", "No"]]
                                 next_name = get_column_name(len(clean_cols) + 1)
                                 current_df[next_name] = ""
                                 st.session_state[ss_key]["df"] = current_df
                                 if hasattr(st, 'rerun'): st.rerun()
                                 else: st.experimental_rerun()
                         
                         with c_c3:
                             pass
                         with c_c4:
                             # Dropdown for column deletion - minimalist
                             clean_cols = [c for c in df.columns if c not in ["_index", "No"]]
                             to_del = st.selectbox("Hapus Kolom", ["Hapus?"] + clean_cols, key=f"sel_del_{file.name}", label_visibility="collapsed")
                             if to_del and to_del != "Hapus?":
                                 current_df = st.session_state[ss_key]["df"]
                                 if to_del in current_df.columns:
                                     current_df.drop(columns=[to_del], inplace=True)
                                     st.session_state[ss_key]["df"] = current_df
                                     if hasattr(st, 'rerun'): st.rerun()
                                     else: st.experimental_rerun()

                    # Display editable table
                    st.markdown("<div class='table-container'>", unsafe_allow_html=True)
                    
                    # Ensure tracking index for consistent row selection
                    if "_index" not in df.columns:
                        df["_index"] = range(len(df))
                        st.session_state[ss_key]["df"] = df

                    # Create Display DataFrame with Row Numbers
                    df_display = df.copy()
                    # Insert visual row number
                    df_display.insert(0, "No", range(1, len(df_display) + 1))
                    
                    # Configuration for AgGrid
                    grid_options = GridOptionsBuilder.from_dataframe(df_display)
                    grid_options.configure_column("_index", hide=True)
                    # Configure "No" column: pinned left, small width, not editable
                    grid_options.configure_column("No", pinned='left', width=50, editable=False, cellStyle={'textAlign': 'center', 'fontWeight': 'bold', 'backgroundColor': '#f0f0f0'})
                    
                    # Auto-size columns based on content
                    for col in df_display.columns:
                        if col in ["No", "_index"]: continue
                        try:
                            # Calculate max length of data (convert to string first)
                            # Using top 100 rows for performance optimization if needed, but for small files full scan is fine.
                            # Given user files seem < 1000 rows, full scan is OK.
                            series = df_display[col].astype(str)
                            max_len = series.map(len).max()
                            if pd.isna(max_len): max_len = 0
                        except Exception:
                            max_len = 0
                        
                        # Compare with header length
                        header_len = len(str(col))
                        target_len = max(max_len, header_len)
                        
                        # Calculate pixel width (approx 12px per char for safety on Cloud fonts + padding)
                        # Increased multiplier from 10 -> 12, padding 30 -> 40
                        width_px = int(target_len * 12) + 40
                        
                        grid_options.configure_column(col, width=width_px)
                    
                    grid_options.configure_default_column(editable=True, enableRowGroup=True)
                    grid_options.configure_selection('multiple', use_checkbox=True)
                    # grid_options.configure_pagination(paginationAutoPageSize=False, paginationPageSize=20) # Disabled as per user request
                    
                    # Apply Search if present
                    if search_term:
                        grid_options.configure_grid_options(quickFilterText=search_term)
                        
                    grid_options = grid_options.build()
                    
                    edited_df = AgGrid(
                        df_display, 
                        gridOptions=grid_options, 
                        enable_enterprise_modules=False, 
                        update_mode=GridUpdateMode.SELECTION_CHANGED | GridUpdateMode.MODEL_CHANGED,
                        data_return_mode=DataReturnMode.AS_INPUT, # Return full data to prevent loss on save when filtered
                        key=grid_key, # Matches logic above
                        height=400,
                        width='100%'
                    )
                    st.markdown("</div>", unsafe_allow_html=True)

                    # --- Editor Formula ---
                    with st.expander("Formula Editor"):
                        st.info("Masukkan formula untuk menghitung nilai kolom. \n"
                                "- **Referensi Sel**: Gunakan format 'A5' untuk merujuk nilai pada kolom A baris 5. Contoh: `A1 + B1`.\n"
                                "- **Operasi Kolom**: Contoh: `A + B` (menjumlahkan seluruh kolom A dan B).\n"
                                "- **Agregasi Baris**: Pilih baris, lalu ketik `SUM`, `AVG`, `MAX`, `MIN`.")
                        
                        f_col1, f_col2 = st.columns([2, 1])
                        
                        # Use original columns relative to data (exclude No and _index)
                        columns = df.columns.tolist()
                        clean_columns = [c for c in columns if c not in ["_index", "No"]]
                        
                        with f_col1:
                            formula_input = st.text_input("Formula", placeholder="Contoh: A5 + B5 atau A + B", key=f"formula_{file.name}")
                        
                        with f_col2:
                            target_col = st.selectbox("Kolom Target (Hasil)", options=clean_columns, key=f"target_{file.name}")
                        
                        if st.button("Terapkan Formula", key=f"apply_formula_{file.name}"):
                            try:
                                # Get latest data from grid
                                # NOTE: 'edited_df['data']' now contains "No" column!
                                grid_data = edited_df['data'].copy()
                                # Drop "No" for logic processing
                                current_data = grid_data.drop(columns=['No'], errors='ignore')
                                
                                selected_rows = edited_df['selected_rows']
                                
                                # Helper for numeric conversion
                                def to_num(s):
                                    return pd.to_numeric(s, errors='coerce').fillna(0)

                                # 1. Determine Scope (Rows)
                                # Check if selected_rows is not empty/None safely
                                has_selection = False
                                if selected_rows is None:
                                    has_selection = False
                                elif isinstance(selected_rows, pd.DataFrame):
                                    has_selection = not selected_rows.empty
                                elif isinstance(selected_rows, list):
                                    has_selection = len(selected_rows) > 0
                                else:
                                    # Fallback for other iterables
                                    has_selection = len(selected_rows) > 0

                                if has_selection:
                                    # If it's a dataframe, convert to list of dicts or handle accordingly
                                    if isinstance(selected_rows, pd.DataFrame):
                                        # Assuming the dataframe has the same structure/columns
                                        selected_indices = selected_rows['_index'].tolist()
                                    else:
                                        # List of dicts
                                        selected_indices = [r.get('_index') for r in selected_rows if r.get('_index') is not None]
                                    
                                    target_mask = current_data['_index'].isin(selected_indices)
                                    row_scope_msg = f"{len(selected_indices)} baris terpilih"
                                else:
                                    target_mask = pd.Series([True] * len(current_data), index=current_data.index)
                                    row_scope_msg = "SEMUA baris (tidak ada seleksi)"
                                    selected_indices = current_data['_index'].tolist()

                                # 2. Check for Aggregation Mode (Row Math)
                                # Trigger if formula is a keyword AND multiple rows selected (or intended generic agg)
                                agg_keywords = {
                                    "SUM": "sum", "+": "sum", "ADD": "sum",
                                    "AVG": "mean", "AVERAGE": "mean",
                                    "MIN": "min",
                                    "MAX": "max"
                                }
                                formula_upper = formula_input.strip().upper()
                                is_aggregation = formula_upper in agg_keywords

                                if is_aggregation and len(selected_indices) > 1:
                                    # --- Aggregation Logic (Row 1 + Row 2 + ...) ---
                                    agg_func = agg_keywords[formula_upper]
                                    
                                    # Get values to agg
                                    values_to_agg = to_num(current_data.loc[target_mask, target_col])
                                    
                                    if agg_func == "sum":
                                        result = values_to_agg.sum()
                                    elif agg_func == "mean":
                                        result = values_to_agg.mean()
                                    elif agg_func == "min":
                                        result = values_to_agg.min()
                                    elif agg_func == "max":
                                        result = values_to_agg.max()
                                    
                                    # Update logic: Where to put the result?
                                    # Standard behavior: Put in the FIRST selected row
                                    first_idx = selected_indices[0]
                                    
                                    # Identify the row in the dataframe that matches this index
                                    # Note: selected_indices are the values of _index column
                                    actual_idx = current_data[current_data['_index'] == first_idx].index
                                    
                                    if not actual_idx.empty:
                                        current_data.loc[actual_idx, target_col] = result
                                        st.toast(f"Hasil agregasi ({result}) disimpan di baris pertama seleksi.")
                                    else:
                                        st.warning("Tidak dapat menemukan baris target untuk menyimpan hasil.")
                                        
                                else:
                                    # --- Column/Cell Logic ---
                                    processed_formula = formula_input
                                    
                                    def replace_cell_ref(match):
                                        col_ref = match.group(1).upper()
                                        row_ref = int(match.group(2))
                                        
                                        if col_ref in clean_columns:
                                            # Lookup by row position (approximate if filtered)
                                            # We grab the tracking _index for safety if needed, 
                                            # but here we use iloc for 'Row N' semantics
                                            idx = row_ref - 1
                                            if 0 <= idx < len(current_data):
                                                val = current_data.iloc[idx][col_ref]
                                                try:
                                                    return str(float(val))
                                                except:
                                                    return str(val)
                                        return match.group(0)

                                    # Pattern: Word boundary, 1+ letters, 1+ digits, Word boundary
                                    processed_formula = re.sub(r'\b([A-Z]+)([0-9]+)\b', replace_cell_ref, processed_formula, flags=re.IGNORECASE)
                                    
                                    # Convert context to numeric
                                    numeric_context = current_data[clean_columns].apply(pd.to_numeric, errors='coerce').fillna(0)
                                    
                                    # Calculate
                                    result_series = numeric_context.eval(processed_formula)
                                    
                                    # Handle scalar
                                    if not isinstance(result_series, pd.Series):
                                        result_series = pd.Series([result_series] * len(current_data), index=current_data.index)
                                    
                                    # Apply
                                    result_subset = result_series.loc[target_mask]
                                    current_data.loc[target_mask, target_col] = result_subset
                                    
                                    st.toast(f"Formula diterapkan pada {row_scope_msg}.")

                                # Update Session State
                                st.session_state[ss_key]["df"] = current_data
                                st.success("Tabel diperbarui!")

                            except Exception as e:
                                st.error(f"Terjadi kesalahan: {e}")
                    

                        

                                

                    
                    # Display edited data preview
                    with st.expander("Lihat Data Hasil Edit (Preview)"):
                        # Clean view without technical columns
                        st.dataframe(edited_df['data'].drop(columns=['_index'], errors='ignore'), use_container_width=True)
                    
                    # Buttons to save edited data
                    st.write("#### Simpan Perubahan")
                    
                    # Prepare clean data for saving (remove _index and No)
                    clean_df = edited_df['data'].drop(columns=['_index', 'No'], errors='ignore')
                    
                    # Use columns for buttons
                    col1, col2 = st.columns(2)
                    with col1:
                        csv = clean_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label=f"Unduh CSV ({file.name})",
                            data=csv,
                            file_name=f"{os.path.splitext(file.name)[0]}_diedit.csv",
                            mime="text/csv",
                            key=f"btn_csv_{file.name}_{i}"
                        )
                    with col2:
                        # Pass the edited data, original content, AND the indices of the D01 lines
                        content, new_filename = save_to_txt(clean_df, original_file_content, d01_indices, file.name)
                        st.download_button(
                            label=f"Unduh TXT ({file.name})",
                            data=content,
                            file_name=new_filename,
                            mime="text/plain",
                            key=f"btn_txt_{file.name}_{i}"
                        )
    
    st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()