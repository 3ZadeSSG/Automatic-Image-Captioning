"""

IMPORTANT: There is no exception handling here in main.py and model.py as Node.js logs the stderr and user only needs to know there was server error
As there might be compatibility issues, resource related errors when pytorch model is loaded, hence only internal server error needs to be shown.

Still exception checks can be added if required.

"""

import argparse
import model
import json

if __name__ == "__main__":

    # Construct the argument parser
    ap = argparse.ArgumentParser()

    # Add the arguments to the parser
    ap.add_argument("-loc", "--image_location", required=True,
    help="Location of Image for caption will be generated")
    args = vars(ap.parse_args())

    imageLocation = str(args['image_location'])

    # Call model to generate caption
    result = model.generateCaptions(imageLocation)
    result = json.dumps(result)
    print(result)

