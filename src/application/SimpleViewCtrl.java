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
import neuralnetworks.NeuralNetworks;
import neuralnetworks.Learning;
import javafx.stage.Stage;

import javax.imageio.ImageIO;

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
	 */
     void save() {
    	String location =NeuralNetworks.location;	
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
		}
		else if(OS.indexOf("nix") >= 0 || OS.indexOf("nux") >= 0 || OS.indexOf("aix") > 0){
			File file = new File(location + "/tmp.png");    			
	    	File fileResized = new File(location + "/tmpResized.png");
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
		}			
    };
    
    /**
     * Make a forward propagation on the canvas and return result
     */
	@FXML
	void analyse() {
		
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
			for (int i=0; i<10; i++){
				System.out.println(results[i]);
			}
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
		gc.setLineWidth(5);
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
