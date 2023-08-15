# Posts
Code used to generate the plots and tables used in the posts.

## Usage

There is a circular reference between these Python scripts. Therefore you must:

1. Run `python professional_programs.py`
2. This will crash, that's okay.
3. Run `python baseline_and_hypothetical_programs.py`
4. Now run `python professional_programs.py` again. It will complete successfully this time.