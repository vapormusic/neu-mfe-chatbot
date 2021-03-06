from pyvi import ViUtils
import underthesea



class InputCleanup:
    def __init__(self):
        print("")

    def word_cleanup(self, input_str):
        #############################################################################
        def isAccented(s):
            try:
                s.encode(encoding='utf-8').decode('ascii')
            except UnicodeDecodeError:
                return True
            else:
                return False

        def filterNames(self, input_str, input_accented):
            namesList = {"SAT"}
            input_str_word = underthesea.word_tokenize(input_str)
            print(input_str_word)
            replacelist = []
            i = 0
            for word in input_str_word:
                if word in namesList:
                    replacelist.append(i)
                i += 1
            print(replacelist)
            accented_str_word = underthesea.word_tokenize(input_accented)
            print(accented_str_word)
            if len(replacelist) != 0:
                for needtoreplace in replacelist:
                    accented_str_word[needtoreplace] = input_str_word[needtoreplace]

            return " ".join(accented_str_word)
        #############################################################################

        output = input_str
        try:
            if not isAccented(input_str):
                accented_input = ViUtils.add_accents(input_str)
                output = filterNames(input_str,accented_input)
            print(output)
        except:
            pass


        return output


