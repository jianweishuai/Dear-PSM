## <p align="center">Dear-PSM：a peptide search engine enables full database search for proteomics</p> 

## Introduction
Dear-PSM, a novel peptide search engine that revolutionizes this approach by supporting full database searching without restricting peptide mass errors and significantly expanding the number of allowable variable modifications per peptide.
Dear-PSM achieves significant breakthroughs:
- Expanded Search Scope: Dear-PSM increases the search range by 40-fold, allowing peptide mass errors from **-6000 to 4500 Daltons**.
- Increased Modification Flexibility: We elevate the maximum allowable modifications per peptide from 3 to 20, expanding potential **modification combinations from 5000 to 1,000,000**.
- High Reproducibility and Novel Discovery: Dear-PSM reproduces over 90% of mainstream search engine results while uncovering a substantial number of new peptides and proteins.
- Enhanced Modification Data Analysis: In phosphorylation data analysis, Dear-PSM increases the **candidate peptide** count from the traditional 140 million to approximately **3 billion**, enabling peptide searches at the scale of billions.
- Improved Performance: Dear-PSM runs **3 to 7 times faster** than mainstream search engines on regular desktops, with **memory consumption reduced by 100 to 240 times**.

## Usage
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
