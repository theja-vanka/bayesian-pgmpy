###################################################
##             ASU CSE 571 ONLINE                ##
##        Unit 3 Reasoning under Uncertainty     ##
##             Project Submission File           ##
##                 burglary.py                   ##
###################################################

###################################################
##                !!!IMPORTANT!!!                ##
##        This file will be auto-graded          ##
##    Do NOT change this file other than at the  ##
##       Designated places with your code        ##
##                                               ##
##  READ the instructions provided in the code   ##
###################################################

# Starting with defining the network structure
from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

def buildBN():

    #!!!!!!!!!!!!!!!  VERY IMPORTANT  !!!!!!!!!!!!!!!
    # MAKE SURE to use the terms "MaryCalls", "JohnCalls", "Alarm",
    # "Burglary" and "Earthquake" as the states/nodes of the Network.
    # And also use "burglary_model" as the name of your Bayesian model.
    ########-----YOUR CODE STARTS HERE-----########
    burglary_model = BayesianModel([('Burglary', 'Alarm'), 
                              ('Earthquake', 'Alarm'),
                              ('Alarm', 'JohnCalls'),
                              ('Alarm', 'MaryCalls')])
    
    cpd_burg = TabularCPD(
        variable='Burglary',
        variable_card=2,
        values=[[1 - 0.001], [0.001]]
    )
    cpd_eq = TabularCPD(
        variable='Earthquake',
        variable_card=2,
        values=[[1 - 0.002], [0.002]]
    )
    cpd_alarm = TabularCPD(
        variable='Alarm',
        variable_card=2,
        values=[
            [1 - 0.001, 1 - 0.94, 1 - 0.29, 1 - 0.95],
            [0.001, 0.94, 0.29, 0.95]
        ],
        evidence=['Earthquake', 'Burglary'],
        evidence_card=[2, 2]
    )
    cpd_jc = TabularCPD(
        variable='JohnCalls',
        variable_card=2,
        values=[
            [1 - 0.05, 1 - 0.9],
            [0.05, 0.9]
        ],
        evidence=['Alarm'],
        evidence_card=[2]
    )
    cpd_mc = TabularCPD(
        variable='MaryCalls',
        variable_card=2,
        values=[
            [1 - 0.01, 1 - 0.7],
            [0.01, 0.7]
        ],
        evidence=['Alarm'],
        evidence_card=[2]
    )

    # Associating the parameters with the model structure.
    burglary_model.add_cpds(cpd_burg, cpd_eq, cpd_alarm, cpd_jc, cpd_mc)

    # Checking if the cpds are valid for the model.
    #print(burglary_model.check_model())
    #print(burglary_model.get_cpds())
    
    ########-----YOUR CODE ENDS HERE-----########
    
    # Doing exact inference using Variable Elimination
    burglary_infer = VariableElimination(burglary_model)

    ########-----YOUR MAY TEST YOUR CODE BELOW -----########
    ########-----ADDITIONAL CODE STARTS HERE-----########
    #print(burglary_infer.query(variables=['JohnCalls'], joint=False, evidence={'Earthquake': 0})['JohnCalls'])
    #print(burglary_infer.query(variables=['MaryCalls'], joint=False, evidence={'Burglary':1, 'Earthquake': 0})['MaryCalls'])
    #print(burglary_infer.query(variables=['MaryCalls'], joint=False, evidence={'Burglary':1, 'Earthquake': 1})['MaryCalls'])
    #print(burglary_infer.query(variables=['MaryCalls'], joint=False, evidence={'JohnCalls': 1})['MaryCalls'])
    #print(burglary_infer.query(variables=['MaryCalls'], joint=False, evidence={'JohnCalls': 1, 'Burglary':0, 'Earthquake': 0})['MaryCalls'])

    ########-----YOUR CODE ENDS HERE-----########
    
    return burglary_infer