## <p align="center">Dear-PSM：a deep learning-based peptide search engine enables full database search for proteomics</p> 

## Introduction
Dear-PSM, a novel peptide search engine that revolutionizes this approach by supporting full database searching without restricting peptide mass errors and significantly expanding the number of allowable variable modifications per peptide.
Dear-PSM achieves significant breakthroughs:
- Expanded Search Scope: Dear-PSM increases the search range by 40-fold, allowing peptide mass errors from **-6000 to 4500 Daltons**.
- Increased Modification Flexibility: We elevate the maximum allowable modifications per peptide from 3 to 20, expanding potential **modification combinations from 5000 to 1,000,000**.
- High Reproducibility and Novel Discovery: Dear-PSM reproduces over 90% of mainstream search engine results while uncovering a substantial number of new peptides and proteins.
- Enhanced Modification Data Analysis: In phosphorylation data analysis, Dear-PSM increases the **candidate peptide** count from the traditional 140 million to approximately **3 billion**, enabling peptide searches at the scale of billions.
- Improved Performance: Dear-PSM runs **3 to 7 times faster** than mainstream search engines on regular desktops, with **memory consumption reduced by 100 to 240 times**.

## Usage
### Dear-PSM for Peptide Search
```
[Small number of files]: 
DearPSM.exe --config=dearpsm.configure --out_dir=/path/to/output_dir/ --input=xx.mzML;yy.mzML;zz.mzML

[Large number of files]:
DearPSM.exe --config=dearpsm.configure --out_dir=/path/to/output_dir/ --input_list=file_list.txt

Introduction of args:
    --config: configure file of DearPSM.
    --out_dir: output directory or path.
    --input: input files. DearPSM supports mzML format. Please use ';' as separators.
    --input_list: input file list. The list contains the paths of spectral file. For example:
                 /path/to/raw1.mzML
                 /path/to/raw2.mzML
                 /path/to/raw3.mzML
                 /path/to/xxxx.mzML

Note: the paths of input files and output directory should not include 'Space' char.
    If there is no dearpsm.configure, please run the following command:
        DearPSM.exe --p
```
### Dear-VIP for Peptide Validation
#### Requirment of python packages
- Numpy
- Pytorch
- pickle

```
[Command line]
python DearVIP-v1.0.0.py

Please enter the path of the Dear-PSM output file below:
/path/to/dearpsm_combine_result.tsv
Please enter the fdr type below, choose "prot" or  "pept":
prot
```
## Configure File
```
# Welcome to use dearpsm v1.0.0 software.
# Development team from Xiamen University, China.
# Our search engine supports full database search 
# and does not need to specify the MS1 tolerance.
# This is the dearpsm parameters file. 
# Everything following the '#' symbol means comment.

search_mode = 1	# Full search=1; Narrow search=0

fasta_name = /path/to/db.fasta

ms1_tol_ppm = 50	# Tolerance of precursor m/z. Note: only use in narrow search! The use can set it to any number in full search.
ms2_tol_ppm = 100	# Tolerance of fragment m/z.
ms1_max_charge = 4	# Precursor max charge.
miss_cleavage = 2	# Miss cleavage of enzyme digestion 
decoy_prefix = rev_	# Prefix of decoy proteins in fasta database
search_enzyme_cutafter = KR	# Residues after which the enzyme cuts.
search_enzyme_skip = P	# Residues that the enzyme will not cut before.

# Up to 20 variable modifications per peptide are supported. 
# format <mass>@<residues>
# Note: 42.0106@n specify represent Acetylation
variable_mod = 42.0106@n
variable_mod = 15.9949@M
#variable_mod = 79.966331@STY
```
