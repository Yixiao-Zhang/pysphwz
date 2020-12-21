
.PHONY: default clean bibtex crop

OUTPUT = output

default : crop

$(OUTPUT) :
	mkdir -p $@

bibtex : $(OUTPUT)/main.aux
	bibtex $^

clean :
	rm -rf $(OUTPUT)

EPS_FIGURES = $(wildcard *.eps)
PDF_FIGURES = $(EPS_FIGURES:.eps=.pdf)

crop: $(PDF_FIGURES)

%.pdf : %.eps
	epstopdf $^ -o $@

# %.pdf : %.uncrop.pdf
# 	pdfcrop $^ $@

# %.uncrop.pdf : %.eps
# 	epstopdf $^ -o $@
