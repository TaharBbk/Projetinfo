package application;

import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.awt.image.RenderedImage;
import java.io.File;
import java.io.IOException;
import java.net.MalformedURLException;
import java.text.DecimalFormat;

import javafx.embed.swing.SwingFXUtils;
import javafx.fxml.FXML;
import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.control.Button;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.image.WritableImage;
import javafx.scene.input.MouseEvent;
import javafx.scene.paint.Color;
import javafx.scene.paint.CycleMethod;
import javafx.scene.paint.RadialGradient;
import javafx.scene.paint.Stop;
import javafx.scene.shape.Circle;
import javafx.scene.shape.Rectangle;
import javafx.scene.text.Text;
import javafx.stage.FileChooser;
import javafx.stage.FileChooser.ExtensionFilter;
import javafx.stage.Stage;

import javax.imageio.ImageIO;

import neuralnetworks.Learning;
import neuralnetworks.NeuralNetworks;

public class SimpleViewCtrl {
	
	NeuralNetworks nN = new NeuralNetworks("best");
	Learning L = new Learning();
	
	@FXML Button boutonAnalyser;
	@FXML Button boutonNouveau;
	@FXML Button boutonImport;
	@FXML Canvas Canvas;
	@FXML Rectangle borders;
	@FXML Circle circle0;
	@FXML Circle circle1;
	@FXML Circle circle2;
	@FXML Circle circle3;
	@FXML Circle circle4;
	@FXML Circle circle5;
	@FXML Circle circle6;
	@FXML Circle circle7;
	@FXML Circle circle8;
	@FXML Circle circle9;
	@FXML Text value0;
	@FXML Text value1;
	@FXML Text value2;
	@FXML Text value3;
	@FXML Text value4;
	@FXML Text value5;
	@FXML Text value6;
	@FXML Text value7;
	@FXML Text value8;
	@FXML Text value9;
	@FXML Text txSuccess;
	@FXML ImageView imageResized;
	
	
	String OS = Learning.OS; 
	
	/**
	 * Save a resized png version of the canvas
	 * @throws IOException 
	 */
     void save() throws IOException {
    	String location = NeuralNetworks.location;	
    	if(OS.indexOf("win") >= 0){
    		File file = new File(location + "\\tmp.png");    			
        	File fileResized = new File(location + "\\tmpResized.png");
        	try {    	            
        		WritableImage writableImage = new WritableImage((int)Canvas.getWidth(), (int)Canvas.getHeight());    	            
        		Canvas.snapshot(null, writableImage);
        		RenderedImage renderedImage = SwingFXUtils.fromFXImage(writableImage, null);
        		ImageIO.write(renderedImage, "png", file);    	             	            
        		BufferedImage resizedImage = new BufferedImage(28, 28, 1);    	    		
        		Graphics2D g = resizedImage.createGraphics();   	    		
        		g.drawImage((BufferedImage)renderedImage, 0, 0, 28, 28, null);    	    		
        		g.dispose();    	    		
        		ImageIO.write(resizedImage, "png", fileResized);    	       
        	 } catch (IOException e) {    	        	
        		 e.printStackTrace();
        	}
		
    	   	BufferedImage image = ImageIO.read(file);
			int hauteur = image.getHeight();
			int largeur = image.getWidth();
			int couleur;
			java.awt.Color color;
			
			double[][] imagetab = new double[hauteur][largeur];			
			
			for (int i=0; i<hauteur; i++){
				for(int j=0; j<largeur; j++){
					color = new java.awt.Color(image.getRGB(i,j), false);
					couleur = (color.getBlue()+color.getRed()+color.getGreen())/3;
					if(couleur<150)
						imagetab[i][j]=1;
					else
						imagetab[i][j]=0;
				}
			}
			
			int xmin = -1, xmax = -1, ymin = -1, ymax = -1;
			
			for (int y=0; y<hauteur; y++){
				boolean lignenulle = true;
				for(int x=0; x<largeur; x++){
					if (ymin == -1 && imagetab[y][x]==1){
						ymin = y;
					}
					if (imagetab[y][x]==1){
						lignenulle = false;
					}
				}
				if (ymin != -1 && ymax == -1 && lignenulle){
					ymax = y;
				}
			}
			
			for (int x=0; x<largeur; x++){
				boolean colonnenulle = true;
				for(int y=0; y<hauteur; y++){
					if (xmin == -1 && imagetab[y][x]==1){
						xmin = x;
					}
					if (imagetab[y][x]==1){
						colonnenulle = false;
					}
				}
				if (xmin != -1 && xmax == -1 && colonnenulle){
					xmax = x;
				}
			}
			
			System.out.println(ymin + ";" + ymax + ";" + xmin + ";" + xmax);
			
			int taille;
			if (ymin-ymax<xmin-xmax){
				taille = xmin-xmax;
			}
			else {
				taille = ymin-ymax;
			}
			
			double[][] imagecentered = new double[taille][taille];
			System.out.println(taille);
			for (int i=0; i<taille; i++){
				for(int j=0; j<taille; j++){
					imagecentered[i][j] = imagetab[i+ymin][j+xmin];
				}
			}
			
			
    	}
    	
    	
    	
    	
    	
		else if(OS.indexOf("nix") >= 0 || OS.indexOf("nux") >= 0 || OS.indexOf("aix") > 0){
			File file = new File(location + "/tmp.png");    			
	    	File fileResized = new File(location + "/tmpResized.png");
	    	try {    	            
	    		WritableImage writableImage = new WritableImage((int)Canvas.getWidth(), (int)Canvas.getHeight());    	            
	    		Canvas.snapshot(null, writableImage);
	    		RenderedImage renderedImage = SwingFXUtils.fromFXImage(writableImage, null);
	    		ImageIO.write(renderedImage, "png", file);    	             	                	       
	    	 } catch (IOException e) {    	        	
	    		 e.printStackTrace();
	    	}
	    
	    	
	    	BufferedImage image = ImageIO.read(file);
			int hauteur = image.getHeight();
			int largeur = image.getWidth();
			int couleur;
			java.awt.Color color;
			
			double[][] imagetab = new double[hauteur][largeur];			
			
			for (int i=0; i<hauteur; i++){
				for(int j=0; j<largeur; j++){
					color = new java.awt.Color(image.getRGB(i,j), false);
					couleur = (color.getBlue()+color.getRed()+color.getGreen())/3;
					if(couleur<150)
						imagetab[i][j]=1;
					else
						imagetab[i][j]=0;
				}
			}

			int xmin = -1, xmax = -1, ymin = -1, ymax = -1;
			
			for (int y=0; y<hauteur; y++){
				boolean lignenulle = true;
				for(int x=0; x<largeur; x++){
					if (ymin == -1 && imagetab[y][x]==1){
						ymin = y;
					}
					if (imagetab[y][x]==1){
						lignenulle = false;
					}
				}
				if (ymin != -1 && ymax == -1 && lignenulle){
					ymax = y;
				}
			}

			for (int x=0; x<largeur; x++){
				boolean colonnenulle = true;
				for(int y=0; y<hauteur; y++){
					if (xmin == -1 && imagetab[y][x]==1){
						xmin = x;
					}
					if (imagetab[y][x]==1){
						colonnenulle = false;
					}
				}
				if (xmin != -1 && xmax == -1 && colonnenulle){
					xmax = x;
				}
			}

			int taille = 0;
			if ((ymax-ymin)<(xmax-xmin)){
				taille = xmax-xmin;
			}
			else {
				taille = ymax-ymin;
			}

			double[][] imagetabcentered = new double[taille][taille];

			for (int i=0; i<taille; i++){
				for(int j=0; j<taille; j++){
					imagetabcentered[i][j] = imagetab[i+ymin][j+xmin];

				}
			}
			try {
			    BufferedImage imagecentered = new BufferedImage(taille, taille, 1);
			    for(int i=0; i<taille; i++) {
			        for(int j=0; j<taille; j++) {
			            int a = (int) imagetabcentered[j][i];
			            java.awt.Color newColor = new java.awt.Color((1-a)*255,(1-a)*255,(1-a)*255);
			            imagecentered.setRGB(j,i,newColor.getRGB());
			        }
			    }
			    File output = new File("imagecentered.png");
			    ImageIO.write(imagecentered, "png", output);
			    BufferedImage resizedImage = new BufferedImage(28, 28, 1);    	    		
	    		Graphics2D g = resizedImage.createGraphics();   	    		
	    		g.drawImage((BufferedImage)imagecentered, 0, 0, 28, 28, null);    	    		
	    		g.dispose();    	    		
	    		ImageIO.write(resizedImage, "png", fileResized);
			}

			catch(Exception e) {
				e.printStackTrace();
			}
		}			
    };
    
    /**
     * Make a forward propagation on the canvas and return result
     * @throws IOException 
     */
	@FXML
	void analyse() throws IOException {
		
		String location = NeuralNetworks.location;
		String nom = "";
		save();
		
		if(OS.indexOf("win") >= 0){
			nom = location + "\\tmpResized.png";
		}
		
		else if(OS.indexOf("nix") >= 0 || OS.indexOf("nux") >= 0 || OS.indexOf("aix") > 0){
			nom = location + "/tmpResized.png";
		}
		
		try {
			double[] image = Learning.imageLecture(nom);
			//image = L.centreReduit(image);
			double[] results = nN.forwardPropagationRAM(image);
			turnOnLights(results);
		} catch (ClassNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		if(OS.indexOf("win") >= 0){
			File file = new File(location + "\\tmp.png");
			File fileResized = new File(location + "\\tmpResized.png");
			file.delete();
			fileResized.delete();
		}
		else if(OS.indexOf("nix") >= 0 || OS.indexOf("nux") >= 0 || OS.indexOf("aix") > 0){
			File file = new File(location + "/tmp.png");
			File fileResized = new File(location + "/tmpResized.png");
			file.delete();
			fileResized.delete();
		}
	}
	
	
	/**
	 * Change color of a circle to identify it
	 */
	void turnOnLights(double[] tab) {
		Text[] values = {value0, value1, value2, value3, value4, value5, value6, value7, value8, value9};
		Circle[] circles = {circle0, circle1, circle2, circle3, circle4, circle5, circle6, circle7, circle8, circle9};
		int imax = NeuralNetworks.max(tab);
		for (int i = 0; i<=9; i++) {
			DecimalFormat numberFormat = new DecimalFormat("#0.00");
			values[i].setText(numberFormat.format(tab[i]));
			Circle c = circles[i];
			c.setRadius(10+10*tab[i]);
			c.setStrokeWidth(0);
			if (i ==imax) {
				RadialGradient gradient1 = new RadialGradient(0,0,0.5,0.5,0.6,true,CycleMethod.NO_CYCLE,
						new Stop(0, Color.RED),
			            new Stop(1, Color.WHITE));
				c.setFill(gradient1);}
			else {
				RadialGradient gradient1 = new RadialGradient(0,0,0.5,0.5,0.5,true,CycleMethod.NO_CYCLE,
						new Stop(0, Color.BLACK),
			            new Stop(1, Color.WHITE));
				c.setFill(gradient1);
				
			}
		}
	}
	
	
	/**
	 * Set back to normal a circle 
	 */
	void turnOffLights() {
		Text[] values = {value0, value1, value2, value3, value4, value5, value6, value7, value8, value9};
		Circle[] circles = {circle0, circle1, circle2, circle3, circle4, circle5, circle6, circle7, circle8, circle9};
		for (int i = 0; i<=9; i++) {
			values[i].setText("");
			Circle c = circles[i];
			c.setRadius(8);
			c.setFill(Color.BLACK);
			c.setStrokeWidth(1);
		}
	}

	
	/**
	 * Enable to draw on the canvas (mouse dragged)
	 * @param event
	 */
	@FXML
	void draw(MouseEvent event) {
		GraphicsContext gc = Canvas.getGraphicsContext2D();
		gc.setLineWidth(12);
        gc.lineTo(event.getX(), event.getY());
        gc.stroke();
	}
	
	
	/**
	 * Enable to move on the canvas (mouse not clicked)
	 * @param event
	 */
	@FXML
	void moveTo(MouseEvent event) {
		GraphicsContext gc = Canvas.getGraphicsContext2D();
        gc.beginPath();
        gc.moveTo(event.getX(), event.getY());
        gc.stroke();
	}
	
	
	/**
	 * Reset canvas and circles
	 */
	@FXML
	void clear() {
		GraphicsContext gc = Canvas.getGraphicsContext2D();
		gc.clearRect(0, 0, Canvas.getWidth(), Canvas.getHeight());
		imageResized.setImage(null);
		turnOffLights();
	}
	
	
	/**
	 * Importe puis analyse une image
	 * @throws MalformedURLException
	 */
	@FXML
	void importation() throws MalformedURLException {
		
		String location = NeuralNetworks.location;
		Stage mainStage = new Stage();

		FileChooser fileChooser = new FileChooser();
		fileChooser.setTitle("Ouvrir l'image a analyser");
		fileChooser.getExtensionFilters().add(new ExtensionFilter("Image Files", "*.png", "*.jpg", "*.gif"));
		File selectedFile = fileChooser.showOpenDialog(mainStage);
		Image apercuImage = new Image("file:" + selectedFile.getAbsolutePath());
		imageResized.setImage(apercuImage);
		
		BufferedImage Image;
		try {
			Image = ImageIO.read(selectedFile);
			
			File fileResized = new File(location + "/tmpResized.png");
	        
	        BufferedImage resizedImage = new BufferedImage(28, 28, 1);
			Graphics2D g = resizedImage.createGraphics();
			g.drawImage(Image, 0, 0, 28, 28, null);
			g.dispose();
			ImageIO.write(resizedImage, "png", fileResized);

			String nom = "";
			
			if(OS.indexOf("win") >= 0){
				nom = location + "\\tmpResized.png";
			}
			
			else if(OS.indexOf("nix") >= 0 || OS.indexOf("nux") >= 0 || OS.indexOf("aix") > 0){
				nom = location + "/tmpResized.png";
			}
			
			try {
				double[] image = Learning.imageLecture(nom);
				double[] results = nN.forwardPropagationRAM(image);
				turnOnLights(results);
			} catch (ClassNotFoundException e) {
				e.printStackTrace();
			} catch (IOException e) {
				e.printStackTrace();
			}
			fileResized.delete();

		  } catch (IOException e) {
	        	e.printStackTrace();
	        }
	}
	
	
	/**
	 * Extraction du taux de succes du reseau
	 */
	@FXML
	void init(){
		txSuccess.setText(nN.getSuccessRate());
	}

}
