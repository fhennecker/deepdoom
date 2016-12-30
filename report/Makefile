all: report.pdf

report.pdf: report.tex report.bib
	latex report
	bibtex report
	latex report
	latex report
	dvips -o report.ps -t letterSize report.dvi
	ps2pdf report.ps report.pdf
