# spTrans (csr2csc): Parallel Transposition of Sparse Data Structures
This repository provides a fast parallel transposition for sparse matrices and graphs on x86-based the multi-core and many-core processors (i.e., CPUs and MICs). The library contains two new parallel transposition algorithms: ScanTrans and MergeTrans. 

* Contact Email: hwang121@vt.edu, kaixihou@vt.edu


## Citing Our Work:
* PlainText:  
Parallel Transposition of Sparse Data Structures. 
Hao Wang, Weifeng Liu, Kaixi Hou, Wu-chun Feng.
In Proceedings of the 30th International Conference on Supercomputing (ICS), 
Istanbul, Turkey, 
June 2016.
* Bibtex:  
@InProceedings{wang-transposition-ics16,  
	author =	{Wang, Hao and Liu, Weifeng and Hou, Kaixi and Feng, Wu-chun},  
	title = 	"{Parallel Transposition of Sparse Data Structures}",  
	booktitle =	{30th International Conference on Supercomputing (ICS)},  
	address =	{Istanbul, Turkey},  
	month =	{June},  
	year =	{2016},  
}

## Usage:
You can make changes to the Makefile accordingly. Especially, you need to enable -DMKL and provide the correct 
MKL installation path if you want to compare MKL results. Examples of setting MKL path are given in the Makefile.
For different vector ISAs, please change the parameter ISA when executing the make command accordingly. Supported ISAs include avx2 (default), mic, avx. The following command compiles the codes using avx2 ISA.
```
$ make ISA=avx2
```
After compilation, run the executable as: (for example, there are 24 threads)
```
$ OMP_NUM_THREADS=24 ./sptrans.out matrixname.mtx
```

## License: 
Please refer to the included LICENSE file.
