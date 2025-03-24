# This is the starting point for developing task-b of the chatbot project.
# Note: Do not start the whole project from this source code!
# Rather, once you have your task-a ready, manage to incorporate it with this source code.
# Notice the incomplete tasks below - indicated with ">>"

#######################################################
#  Initialise NLTK Inference
from nltk.sem import Expression
from nltk.inference import ResolutionProver
import pandas as pd
import aiml

read_expr = Expression.fromstring

# Load Knowledge Base (handling comma issues)
kb = []

try:
    data = pd.read_csv('logical-kb.csv', header=0, names=["Fact", "Type"], delimiter=",", quotechar='"')

    for index, row in data.iterrows():
        fact = row["Fact"]
        expr = read_expr(fact)
        kb.append(expr)

    print("Knowledge base loaded successfully!")

except Exception as e:
    print(f"Error loading KB: {e}")
    exit(1)  # Exit if the KB fails to load


#  Initialise AIML agent
kern = aiml.Kernel()
kern.setTextEncoding(None)

try:
    kern.bootstrap(learnFiles="mybot-logic.xml")
    print("AIML bot initialized successfully!")
except Exception as e:
    print(f"Error initializing AIML bot: {e}")
    exit(1)

# Welcome user
print("Welcome to this chat bot. Please feel free to ask questions from me!")

# Main loop
while True:
    try:
        userInput = input("> ")
    except (KeyboardInterrupt, EOFError):
        print("Bye!")
        break

    responseAgent = 'aiml'

    if responseAgent == 'aiml':
        answer = kern.respond(userInput)

    # Post-process the answer for commands
    if answer and answer[0] == '#':
        params = answer[1:].split('$')
        cmd = int(params[0])

        if cmd == 0:
            print(params[1])
            break

        elif cmd == 31:  # "I know that * is *"
            object, subject = params[1].split(' is ')
            expr = read_expr(f"{subject}({object})")

            # Check for contradictions before adding
            if ResolutionProver().prove(expr.negate(), kb):
                print(f"Error: '{object} is {subject}' contradicts existing knowledge!")
            else:
                kb.append(expr)
                print(f"OK, I will remember that {object} is {subject}")

        elif cmd == 32:  # "Check that * is *"
            object, subject = params[1].split(' is ')
            expr = read_expr(f"{subject}({object})")

            answer = ResolutionProver().prove(expr, kb, verbose=False)

            if answer:
                print("Correct.")
            else:
                # Check if the negation is provable
                negated_expr = expr.negate()
                negation_result = ResolutionProver().prove(negated_expr, kb, verbose=False)

                if negation_result:
                    print("Incorrect.")
                else:
                    print("Sorry, I don't know.")

        elif cmd == 99:
            print("I did not get that, please try again.")

    else:
        print(answer)
