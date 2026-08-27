all: thesis.pdf

thesis.pdf: main.tex solutionclass.cls bib.bib bildverzeichnis.bib
	latexmk -pdf -interaction=nonstopmode main.tex
	cp main.pdf thesis.pdf

clean:
	latexmk -C main.tex
	rm -f thesis.pdf

.PHONY: all clean
