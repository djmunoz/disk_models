import time, sys

# update_progress() : Displays or updates a console progress bar
## Accepts a float between 0 and 1. Any int will be converted to a float.
## A value under 0 represents a 'halt'.
## A value at 1 or bigger represents 100%
def update_progress(completed,total):
    progress = (completed + 1) * 1.0/total
    barLength = 20 # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength*progress))
    text = "\rCalculating: [{0}] {1:.2f}% {2}".format( "="*block + " "*(barLength-block), progress*100, status)
    if (total-completed == 1): text = text+"\n"
    sys.stdout.write(text)
    sys.stdout.flush()
