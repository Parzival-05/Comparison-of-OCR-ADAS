class OCRPostprocess:
    def __init__(self, gtd: list[list[str]]):
        self.chars = set()
        for words in gtd:
            for word in words:
                for char in word.lower():
                    self.chars.update(char)

    def remove_excess_chars(self, string: str):
        return "".join(char for char in string if char in self.chars)

    def postprocess(self, result: list[str]):
        postprocessed = []
        for string in result:
            postprocessed_string = self.remove_excess_chars(string)
            if postprocessed_string:
                postprocessed.append(postprocessed_string)
        return postprocessed
