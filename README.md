## <p align="center">Dear-PSM: a deep learning-based peptide search engine enables full database search for proteomics</p> 

## Introduction
Dear-PSM, a peptide search engine that supports full database searching. It does not restrict peptide mass errors, matching each spectrum to all peptides in the database and increasing the number of variable modifications per peptide from the conventional 3 to 20. 

## Usage
```
"[Small number of files]: "
"DearPSM.exe --config=dearpsm.configure --out_dir=/path/to/output_dir/ --input=xx.mzML;yy.mzML;zz.mzML\n"
"[Large number of files]: "
"DearPSM.exe --config=dearpsm.configure --out_dir=/path/to/output_dir/ --input_list=file_list.txt\n"
"Introduction of args:\n"
"    --config: configure file of DearPSM.\n"
"    --out_dir: output directory or path.\n"
"    --input: input files. DearPSM supports mzML format. Please use ';' as separators.\n"
"    --input_list: input file list. The list contains the paths of spectral file. For example:\n"
"                 /path/to/raw1.mzML\n"
"                 /path/to/raw2.mzML\n"
"                 /path/to/raw3.mzML\n"
"                 /path/to/xxxx.mzML\n\n"
"Note: the paths of input files and output directory should not include 'Space' char.\n"
"    If there is no dearpsm.configure, please run the following command:\n"
"        DearPSM.exe --p\n\n"
```
