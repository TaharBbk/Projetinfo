import java.awt.Color;
import java.io.IOException;

public class Test {
	int nombre;
	
	// fonction qui doit renvoyer en sortie un nombre compris entre 0 et 9 de manière "aléatoire"
	public int test(){
		double a = Math.random();
		nombre = (int)(a*10);		
		return nombre;
		}
	
	// fonction qui renvoie un nombre en suivant une loi particulière
	public int test1(){
		double a = Math.random();
		if(a<0.1){
			return 0;
		}
		if (0.1<=a && a<0.2){
			return 1;
		}
		if (0.2<=a && a<0.3){
			return 2;
		}
		if (0.3<=a && a<0.4){
			return 3;
		}
		if (0.4<=a && a<0.5){
			return 4;
		}
		if (0.5<=a && a<0.6){
			return 5;
		}
		if (0.6<=a && a<0.7){
			return 6;
		}
		if (0.7<=a && a<0.8){
			return 7;
		}
		if (0.8<=a && a<0.9){
			return 8;
		}
		else{
			return 9;
		}
		}
	
	public void learning(){
		for (int i=0; i<2000; i++){
			for (int j=0; j<10; i++){
				String nom = j + "_0" + i ;
				try {
					backPropagation(nom,j);
				} catch (ClassNotFoundException e) {
					e.printStackTrace();
				} catch (IOException e) {
					e.printStackTrace();
				}
				// prendre les 1/3 images
				// effectuer un test
				//modifier la matrice des poids
			}
			}	
		}
	
	public double test(int cacher){
		int reussit;
		int result;
		for (int i=0; i<2000; i++){
			int premier = 2000 + i;
			for (int j=0; j<10; i++){
				String nom = j + "_0" + i ;
				result = forwardPropagation(nom);
				if (result == j){
					reussit = reussit +1 ;
				}
			}
			TO DO 
			faire varier la couche cacher dans le test car la, pas utiliser
			
			// prendre les 1/3 images
			// effectuer un test avec une couche de neurones cacher de valeur "cacher"
			// rendre le taux de succès
			}
		return (double)((double)reussit/(double)20000); 
		}
	
	public double findtherightone(int range){
		
		
		//lancer learning
		// faire varier "cacher" puis lancer "test" plusieurs fois 
	}

}
