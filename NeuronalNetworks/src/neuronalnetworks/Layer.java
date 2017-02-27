package neuronalnetworks;

public class Layer {

	double[] values;
	double[][] weights;
	private Layer Null;
	Layer next = Null;
	
	//Constructeur 1
	public Layer(double[] val, double[][] weight, Layer next){
		this.values = val;
		this.weights = weight;
		this.next=next;
	}
	
	//Constructeur 2
	public Layer(double[] val, double[][] weight){
		this.values = val;
		this.weights = weight;
	}
	
	public Layer() {}
	
	public double[][] getWeights() {
		return weights;
	}

	public void setWeights(double[][] weights) {
		this.weights = weights;
	}

	public void setValues(double[] values) {
		this.values = values;
	}

	//Recuperation des valeurs des neurones de la couche
	public double[] getValues(){
		return this.values;
	}
	
	//Fonction d'activaton
	public void activate(){
		activationFunction(this.values);
	}
	
	//Foncion de propagation du reseau de neurone
	public void propagate(){
		if(this.next!=Null){
			this.next.execute();
		}
		this.next.setValues(productMatrix(this.values, this.weights));
	}
	
	//Execution de la forward propagation
	public void execute(){
		this.activate();
		this.propagate();
	}
	
	//Produit Matriciel entre un vecteur et une matrice
	public double[] productMatrix(double[] MB, double[][] MA){
		int ha = MA.length;
		int la = MA[0].length;
		int hb = MB.length;
		assert(la==hb);
		double[] produit = new double[ha];
		
		for(int i=0; i<ha; i++){
			int sum = 0;
			for(int h=0; h<hb; h++){
				sum+=MA[i][h]*MB[h];
			}
			produit[i]=sum;
		}
		
		return produit;
	}
	
	//Implementation de la fonction 
	public void activationFunction(double[] M){
		int ha = M.length;
		for(int i=0; i<ha; i++){
			M[i]=(1/(1+Math.exp(-1*M[i])));
		}
	}
}
