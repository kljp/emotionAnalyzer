from konlpy.tag import Twitter

# del_word = '!$?'
# del_line = '@%#'
#
# docs_input = open('input.txt', 'r')
# docs_output = open('output.txt', 'w')
# for doc in docs_input:
#     twit = Twitter().pos(doc)
#     for i in range(0, len(twit)):
#         if twit[i][1] == 'Noun' or twit[i][1] == 'Adjective':
#             docs_output.write(twit[i][0] + del_word)
#     docs_output.write(del_line)
#
# docs_input.close()
# docs_output.close()



docs_input = open('input_test.txt', 'r')
docs_output = open('output_test.txt', 'w')
for doc in docs_input:
    twit = Twitter().pos(doc)
    for i in range(0, len(twit)):
        if twit[i][1] == 'Noun' or twit[i][1] == 'Adjective' or twit[i][1] == 'Verb':
            docs_output.write(twit[i][0] + ' ')
    docs_output.write('\n')

docs_input.close()
docs_output.close()

