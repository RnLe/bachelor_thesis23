all: manuscript.pdf

manuscript.pdf: main.tex references.bib
	latexmk -pdf -interaction=nonstopmode main.tex
	cp main.pdf manuscript.pdf

clean:
	latexmk -C main.tex
	rm -f manuscript.pdf

.PHONY: all clean
