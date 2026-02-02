# IDP Final Project

> Add instructions here to explain the all the folders & files in this project.
> 
> This file should be the starting point with enough directions and context that someone that has no idea what you have done or how to run your project, can get your code and work out how to view the documentation and run your project.
> 
> See the [NCHS IDP Website](https://nchs-cs.github.io/notes/project/) for the latest directions.
>
> You may add jupyter notebooks instead of the python files if you like, but use the same names as the template python files given.

---

## Phase 1: Discovery Document + Raw Data

**Due: Mon, Feb 23, 2026**
[https://nchs-cs.github.io/notes/project/](https://nchs-cs.github.io/notes/project/)

### Where this lives in your repository

* **`discovery_raw_data/README.md`**
  This folder is where you place:

  * your `discovery.md` document
  * your **raw, unprocessed data** (or a Google Drive link if the dataset is larger than ~1GB)

### How this connects to the project requirements

In Phase 1, you are showing that you understand:

* what question(s) you are trying to answer and why they matter
* what data you are using and where it comes from
* what challenges or limitations you expect to encounter

This folder is the **home for your Phase 1 submission**: the written discovery document plus the raw data (or links).

---

## Phase 2: Data Organization

**Due: Mon, Mar 16, 2026**
[https://nchs-cs.github.io/notes/project/](https://nchs-cs.github.io/notes/project/)

### Where this lives in your repository

* **`data_organized/README.md`**
  This folder contains data that is:

  * cleaned, reduced, and normalized
  * small enough to load quickly and repeatedly
  * ready to be plotted

  You should also explain how to run your preprocessing code locally.

* **`data_organized/sketches/README.md`**
  This folder contains:

  * image-only sketches of your planned plots
  * a table that lists each sketch, links to the image, and explains how it helps answer one or more research questions

* **`data_organized/test_data/README.md`**
  Use this folder if you:

  * create small test datasets to verify your preprocessing logic
  * want to test plots or transformations using fake or simplified data

### How this connects to the project requirements

Phase 2 is about proving that you can **prepare data for analysis**:

* reduce raw data into a usable form
* partially or fully organize it using code
* plan your plots before implementing them
* optionally include unit tests or advanced data sourcing as challenge goals

Everything in `data_organized/` exists to make Phase 3 faster, cleaner, and repeatable.

---

## Phase 3: Final Code + Research Document

**Due: Fri, May 1, 2026**
[https://nchs-cs.github.io/notes/project/](https://nchs-cs.github.io/notes/project/)

### Where this lives in your repository

* **`final_report/README.md`**
  This folder is where you:

  * save all generated plot images (created by `create_plots.py`)
  * place your written research report (`final.md`)
  * document each plot and explain how it answers specific research question(s)

* **Root `README.md`**
  This is the **front door to your project**. Someone who knows nothing about your work should be able to:

  * understand the purpose of your project
  * see how the folders are organized
  * follow instructions to run your code and read your report

### What “done” looks like for Phase 3

By the end of Phase 3, someone should be able to:

* run your preprocessing code to generate organized data
* run your plotting code to generate visualizations
* open `final_report/final.md` and read a clear, plot-driven analysis
* use your README files to understand how everything fits together

---

## Phase 4: Final Presentation

**Presentation weeks: May 18 / May 25 / June 1, 2026**
[https://nchs-cs.github.io/notes/project/](https://nchs-cs.github.io/notes/project/)

### Where this comes from in your repository

There is no dedicated `presentation/` folder, but your presentation is built directly from:

* the plots generated in **`final_report/`**
* the explanations you wrote in **`final_report/README.md`** and `final.md`

### How this connects to the project requirements

Your presentation should be **plot-driven**:

* most slides should be visualizations
* your spoken explanation provides context, interpretation, and conclusions

Phase 4 is essentially you **telling the story of your project** using the artifacts you created in Phase 3.

---

## One-screen checklist (what you need to complete in each README)

* **Root `README.md`**
  Explain what your project is, how the repo is organized, and how to run it.

* **`discovery_raw_data/README.md`**
  Point to `discovery.md` that describes and links to your origin for raw data sources.

* **`data_organized/README.md`**
  Explain how your data was cleaned, reduced, and prepared.

* **`data_organized/sketches/README.md`**
  Include a table linking each sketch to a research question.

* **`data_organized/test_data/README.md`**
  Describe any test datasets and what they are used to validate.

* **`final_report/README.md`**
  Include a table linking each final plot to a research question and explain how to generate the plots.

