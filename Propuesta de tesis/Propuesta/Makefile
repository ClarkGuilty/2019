all: propuesta_de_tesis.pdf clean


propuesta_de_tesis.pdf: mybib.bib propuesta_de_tesis.tex
	pdflatex propuesta_de_tesis.tex
	biber propuesta_de_tesis
	pdflatex propuesta_de_tesis.tex
	pdflatex propuesta_de_tesis.tex


clean:
	rm *.aux *.bbl *.bcf *.blg *.log *.out *.run.xml 
