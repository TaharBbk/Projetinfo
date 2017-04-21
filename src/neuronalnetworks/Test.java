package neuronalnetworks;
import java.io.IOException;

public class Test {
	public double[][][] betterweights;
	double bestSuccessRate;
	
	// fonction qui doit renvoyer en sortie un nombre compris entre 0 et 9 de maniÃ¨re "alÃ©atoire"
	public int uniform(){
		double a = Math.random();
		return (int)(a*10);
	}
	
	public void learning(NeuronalNetworks N){
		for (int i=0; i<2000; i++){
			for (int j=0; j<10; j++){
				String nom = j + "_0" + i ;
				try {
					N.backPropagation(nom,j);
				} catch (ClassNotFoundException e) {
					e.printStackTrace();
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
		}
	}
	
	public double test(NeuronalNetworks N){
		double reussit = 0;
		double[] result;
		for (int i=2000; i<4000; i++){
			for (int j=0; j<10; i++){
				String nom = j + "_0" + i ;
				try {
					result = N.forwardPropagation(nom);
					if (result[j] == 1){
						reussit = reussit +1 ;
					}
				} catch (ClassNotFoundException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				} catch (IOException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
			}
		}
		return reussit/20000; 
	}
	
	public void findtherightone(){
		for(int i=492; i<792; i++){
			NeuronalNetworks N = new NeuronalNetworks(i);
			learning(N);
			double successRate = test(N); 
			if(successRate>bestSuccessRate){
				betterweights=N.weights;
				bestSuccessRate=successRate;
			}
		}
	}
}
