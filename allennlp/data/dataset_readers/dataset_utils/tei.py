import re
import spacy

TEI2IOB = {"persName": "PER", "orgName": "ORG", "placeName": "LOC"}
START_SUFFIX = "-start"
END_SUFFIX = "-end"


def strip_suffix(string):
    return string.split("-")[0]


def annotate(xml):
    # xml matches the pattern above
    if xml[1] == "/":
        return xml[2:-1] + END_SUFFIX

    else:
        return xml[1:-1] + START_SUFFIX


def strip_word(word, matches, nlp, stripped, all_tokens, no_space, annotations):
    pattern_start = re.compile("<[a-zA-Z_]+>")
    pattern_end = re.compile("</[a-zA-Z_]+>")
    w_annotations = []
    for match in matches:
        w_annotations.append(annotate(match))
    splitted_start = re.split(pattern_start, word)
    # TODO: we assume no word contains more than one annotation
    if len(splitted_start) > 1:
        prefix, rest = splitted_start
        if prefix:
            tokens = list(nlp(prefix))
            all_tokens.extend(tokens)
            # The prefix requires space before, but the tag itself not
            no_space[len(stripped) + 1] = True
            stripped.append(prefix)
    else:
        rest = splitted_start[0]
    splitted_end = re.split(pattern_end, rest)
    tag = splitted_end[0]
    stripped.append(tag)
    tokens = list(nlp(tag))
    n_tokens = len(all_tokens)
    for j, _ in enumerate(tokens):
        annotations[n_tokens + j] = w_annotations
    all_tokens.extend(tokens)
    if len(splitted_end) > 1:
        suffix = splitted_end[1]
        if suffix:
            tokens = list(nlp(suffix))
            all_tokens.extend(tokens)
            no_space[len(stripped)] = True
            stripped.append(suffix)


def split_annotations(txt, nlp):
    pattern = re.compile("</?[a-zA-Z_]+>")
    original_words = txt.split()
    stripped = []
    # A mapping between token index and its annotations
    annotations = {}
    all_tokens = []
    # A mapping between stripped_words index and
    # whether it's preceded by a space
    no_space = {}
    for word in original_words:
        matches = re.findall(pattern, word)
        if matches:
            strip_word(word, matches, nlp, stripped, all_tokens, no_space, annotations)
        else:
            stripped.append(word)
            tokens = list(nlp(word))
            all_tokens.extend(tokens)

    return (stripped, annotations, no_space, len(all_tokens))


def reassemble_txt(stripped_words, no_space):
    stripped_txt = stripped_words[0]
    for i, word in enumerate(stripped_words[1:]):
        if i + 1 in no_space:
            stripped_txt += word
        else:
            stripped_txt += " " + word

    return stripped_txt




def convert_annotations_to_iob(annotations, max_tokens, mapping=TEI2IOB):
    result = []
    state = "out"  # possible values: out, start, mid, end
    tag = "O"
    for ind in range(max_tokens):
        if ind in annotations:
            anno = annotations[ind]
            is_start = any(an.endswith("start") for an in anno)
            is_end = any(an.endswith("end") for an in anno)
            if is_start:
                if "out" == state:
                    tag = "I"
                elif "end" == state:
                    tag = "B"
                state = "start"
                # note it can have both start and end tags
                if is_end:
                    state = "end"
            else:  # must be end without start
                state = "end"
                tag = "I"
            tag += '-' + mapping[strip_suffix(anno[0])]

        elif "start" == state:
            state = "mid"
            # tag could be 'I-XXX' or 'B-XXX'
            tag = "I" + tag[1:]

        elif "end" == state:
            state = "out"
            tag = "O"

        # if there are no annotations, and state is 'mid' or 'out',
        # there are no changes, state and tag stay the same
        result.append(tag)

    return result


def tei_to_iob(txt):
    nlp = spacy.load("en")
    words, annotations, no_space, n_tokens = split_annotations(txt, nlp)
    stripped_txt = reassemble_txt(words, no_space)
    tags = convert_annotations_to_iob(annotations, n_tokens)
    doc = nlp(stripped_txt)

    return (doc, annotations, tags)


def main():
    txt = "<persName>Harry Potter</persName> goes to \
        <orgName>Hogwarts</orgName>. <persName>Sally</persName> \
        lives in #<placeName>London</placeName> \
        <placeName>England</placeName> <placeName>UK</placeName>."
    doc, annotations, tags = tei_to_iob(txt)
    n_tokens = 0
    print(txt)
    for sent_ind, sent in enumerate(doc.sents):
        print("sentence{}: {}".format(sent_ind, sent))
        for tok in list(sent):
            if n_tokens in annotations:
                anns = annotations[n_tokens]
            else:
                anns = []
            print(
                "\t token{}: {}, annotations: {} tag: {}".format(
                    n_tokens, tok, anns, tags[n_tokens]
                )
            )
            n_tokens += 1


if __name__ == "__main__":
    main()
