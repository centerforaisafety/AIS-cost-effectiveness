# AI safety cost-effectiveness

This repository contains code for estimating the cost-effectiveness of various field-building programs. It has been built for the [Center for AI Safety](https://www.safe.ai/).

## Using this repository

To get this repository working on your local machine:

1. Install Python and your preferred code editor.
2. Fork this repository.
3. Install the repository's dependencies by executing `pip install -r requirements.txt` in your terminal.

Then, see the [examples README](/src/scripts/examples/README.md) for demonstrations of the repository's use.

If you would like assistance with this repo and/or your own evaluations, contact CAIS at [contact@safe.ai](mailto:contact@safe.ai).

## Directory Structure

- `src`: Contains source code.
    - `models`: Contains cost-effectiveness models, the main logic of this project.
    - `parameters`: Contains parameter instances for each program evaluated. (The [parameters README](/src/parameters/README.md) describes what we mean by instances.)
    - `scripts`: Contains scripts for generating outputs. Organized into subdirectories for `examples` of how to use the repository, and code used to generate content for written `posts`.
    - `utilities`: Contains functions and assumptions that are common across multiple cost-benefit analyses. Organized into subdirectories for `assumptions`, `defaults`, `functions`, `plotting`, and `sampling`.
- `output`: Contains data and plot outputs generated from `scripts`.

The `scripts` feed `parameters` into `models` to produce `output`s. The `utilities` are used at many different stages of the project -- providing functions for specifying parameters, lower-level functions for models, sampling functions for the scripts, plotting functions for the outputs, and more.

## Other resources connected to this project

Our [introduction post]() lays out our approach to modeling – including our motivations for using models, the benefits and limitations of our key metric, comparisons between programs for students and professionals, and more.

We have two posts evaluating [student programs]() and [professional programs]() respectively.

Finally, for definitions and values of the parameters used in our models, refer to the [Parameter Documentation](https://docs.google.com/spreadsheets/d/1uK4opqsCmC5nW6G3D1X67KZnQdMGGL1YbpVmQN5OKF8/edit#gid=581108234) sheet.

## Programs

### Evaluated in this repository

1. The [**Trojan Detection Challenge**](https://trojandetection.ai/) (or ‘**TDC**’): A prize at a top ML conference.
2. The [**NeurIPS ML Safety Social**](https://www.mlsafety.org/social) (or ‘**NeurIPS Social**’): A social at a top ML conference.
3. The [**NeurIPS ML Safety Workshop**](https://neurips2022.mlsafety.org/) (or ‘**NeurIPS Workshop**’): A workshop at a top ML conference.
4. The [**Atlas Fellowship**](https://www.atlasfellowship.org/): A 10-day in-person program providing a scholarship and networking opportunities for select high school students.
5. [**ML Safety Scholars**](https://forum.effectivealtruism.org/posts/9RYvJu2iNJMXgWCBn/introducing-the-ml-safety-scholars-program) (or '**MLSS**'): CAIS’s discontinued summer course, designed to teach undergraduates ML safety.
6. **Student Club** : A high-cost, high-engagement student club at a top university, similar to [HAIST, MAIA](https://www.lesswrong.com/posts/LShJtvwDf4AMo992L/update-on-harvard-ai-safety-team-and-mit-ai-alignment), or [SAIA](https://www.lesswrong.com/posts/zgJCSK5KdkiKDuuCw/the-tree-of-life-stanford-ai-alignment-theory-of-change).
7. **Undergraduate Stipends**: Specifically, the [ML Safety Student Scholarship](https://www.mlsafety.org/safety-scholarship), which provides stipends to undergraduates connected with research opportunities.

### Not evaluated in this repository

Notice that the [`professional_program`](src/models/professional_program.py) and [`student_program`](src/models/student_program.py):

1. Are flexible enough to accommodate a wide range of possible field-building programs, and
2. Could be easily repurposed for research areas beyond AI safety.

We hope that the tools in this repository might be used or extended by other organizations. For suggestions of how to go about this, see the [examples README](/src/scripts/examples/README.md).

## License

This project is licensed under the [MIT License](https://en.wikipedia.org/wiki/MIT_License).
