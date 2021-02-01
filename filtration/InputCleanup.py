from pyvi import ViUtils




class InputCleanup:
    def __init__(self):
        print("")

    def word_cleanup(self, input_str):
        output = input_str.lower()
        output2 = output
        try:
            output2 = ViUtils.add_accents(output)
        except:
            pass

        return output2


