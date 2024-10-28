import numpy as np
import itertools

class TurnEnvironment:
    def __init__(self,nb_dices,nb_faces):
        self.dices = nb_dices
        self.faces = nb_faces
        #the state representation is the histogram of the face values
        self.Roll, self.Roll_P = self.get_states()
        #compute state space and probability for all subsets of dices to pre-compute reroll probabilities
        self.S = self.Roll[0]
        self.Aa = self.get_actions_list()
        
        
    def get_states(self):
    #return states and associated probability for all subsets of dices for reroll to pre-compute reroll probabilities
        Roll = [ [] for _ in range(self.dices) ]
        Roll_P = []
    
        # For 1 to self.dices, generate all possible states of n dices
        for d in range(self.dices):
            for it in itertools.product(range(self.faces),repeat=self.dices):
                s = np.bincount(np.array(it[d:], dtype='int'),minlength=self.faces)
                Roll[d].append(s) 
      
        for  d in range(self.dices):
            S,counts = np.unique(Roll[d],axis=0,return_counts=True)
            Roll[d] = list(S)
            Roll_P.append(counts/counts.sum())
        return Roll, Roll_P
    
    def get_actions_from_state(self,s) :
        # action are described as keeped dices histogram (the reroll number is implicit)
        nb_actions = np.prod(s+1)
        A = np.zeros((nb_actions,self.faces),dtype='int')   
        # first step is to generate the list of iterables
        l = []
        for f in range(self.faces) :
            l.append(list(np.arange(s[f]+1)))
        k=0
        # then generates all possible actions
        for i in itertools.product(*l): 
            A[k,:]=np.array(i) 
            k = k+1
        return A

    def get_actions_list(self):
        Aa = []
        for s in self.S:
            Aa = Aa + self.get_actions_from_state(s).tolist()
            Aa = list(np.unique(Aa,axis=0))
        return Aa


    def get_state_index(self,S):
          return np.argwhere((self.S == S).sum(axis=1)==self.faces)[0][0]

    def get_action_index(self,a):
         return np.argwhere((a == self.Aa).sum(axis=1)==self.faces)[0][0]

    def get_states_from_action(self,a) :
        if a.sum() == self.dices :
             return [a],np.array([1.])
        return  np.array(self.Roll[a.sum()])+a,self.Roll_P[a.sum()]


    # BACKWARD RECURSION
    def One_step_backward(self,v_out):
         v_in = np.zeros(len(self.S))
         a_in = np.zeros(len(self.Aa))
         Q_in = np.zeros((len(self.S),len(self.Aa)))

         for a in self.Aa:
             i_a = self.get_action_index(a)
             Sr,Pr = self.get_states_from_action(a)
             k = 0
             for k in range(len(Pr)) :
                 i_sr = self.get_state_index(Sr[k])
                 a_in[i_a]+=  Pr[k]*v_out[i_sr]
                 k=k+1
                 #########################"
         for s in self.S :
             A = self.get_actions_from_state(s)
             i_s = self.get_state_index(s)
             for a in A :
                 i_a = self.get_action_index(a)
                 Q_in[i_s][i_a] = a_in[i_a]
                 v_in[i_s] = Q_in[i_s].max()
    
         return v_in, Q_in 
     
        
    def get_state_from_action(self,a) :
        #reroll number
        r = self.dices - a.sum()
        dice_view = list(np.random.randint(0,high=self.faces,size=r))
        s = np.zeros((self.faces),dtype='int')
        for f in range(self.faces) :
            s[f] = dice_view.count(f)
            if s.sum() == r :
                break
        return a+s

    def choose_best_action(self,s,Q):
        i_s = self.get_state_index(s)
        i_a = np.argmax(Q[i_s])
        return self.Aa[i_a],i_a

    
    
if __name__ == '__main__':
    env = TurnEnvironment(3, 4)
    print(env.S)
    