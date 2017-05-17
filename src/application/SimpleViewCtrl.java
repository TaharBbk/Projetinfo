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

import neuronalnetworks.NeuronalNetworks;

public class SimpleViewCtrl {
	
	NeuronalNetworks nN = new NeuronalNetworks(492,true);
	
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
	
	//save a resised png version of the canvas
    void save() {
    	String location = NeuronalNetworks.location;
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
    };
	
    //make a forward propagation on the canvas and return result
	@FXML
	void analyse() {
		
		String location = NeuronalNetworks.location;
		save();
		
		try {
			double[] results = nN.forwardPropagationRAM(NeuronalNetworks.imageLecture("tmpResized"));
			turnOnLights(results);
		} catch (ClassNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		File file = new File(location + "/tmp.png");
		File fileResized = new File(location + "/tmpResized.png");
		file.delete();
		fileResized.delete();
	}
	
	//change color of a circle to identify it
	void turnOnLights(double[] tab) {
		Text[] values = {value0, value1, value2, value3, value4, value5, value6, value7, value8, value9};
		Circle[] circles = {circle0, circle1, circle2, circle3, circle4, circle5, circle6, circle7, circle8, circle9};
		int imax = NeuronalNetworks.max(tab);
		for (int i = 0; i<=9; i++) {
			DecimalFormat numberFormat = new DecimalFormat("#0.00");
			values[i].setText(numberFormat.format(tab[i]));
			Circle c = circles[i];
			c.setRadius(10+10*tab[i]);
			c.setStrokeWidth(0);
			if (i ==imax) {
				RadialGradient gradient1 = new RadialGradient(0,0,0.5,0.5,0.6,true,CycleMethod.NO_CYCLE,
						new Stop(0, Color.DARKRED),
			            new Stop(1, Color.WHITE));
				c.setFill(gradient1);}
			else {
				RadialGradient gradient1 = new RadialGradient(0,0,0.5,0.5,0.5,true,CycleMethod.NO_CYCLE,
						new Stop(0, Color.RED),
			            new Stop(1, Color.WHITE));
				c.setFill(gradient1);
				
			}
		}
	}
	
	//set back to normal a circle 
	void turnOffLights() {
		Text[] values = {value0, value1, value2, value3, value4, value5, value6, value7, value8, value9};
		Circle[] circles = {circle0, circle1, circle2, circle3, circle4, circle5, circle6, circle7, circle8, circle9};
		for (int i = 0; i<=9; i++) {
			values[i].setText("");
			Circle c = circles[i];
			c.setRadius(8);
			c.setFill(Color.RED);
			c.setStrokeWidth(1);
		}
	}

	//enable to draw on the canvas (mouse dragged)
	@FXML
	void draw(MouseEvent event) {
		GraphicsContext gc = Canvas.getGraphicsContext2D();
		gc.setLineWidth(5);
        gc.lineTo(event.getX(), event.getY());
        gc.stroke();
	}
	
	//enable to move on the canvas (mouse not clicked)
	@FXML
	void moveTo(MouseEvent event) {
		GraphicsContext gc = Canvas.getGraphicsContext2D();
        gc.beginPath();
        gc.moveTo(event.getX(), event.getY());
        gc.stroke();
	}
	
	//reset canvas and circles
	@FXML
	void clear() {
		GraphicsContext gc = Canvas.getGraphicsContext2D();
		gc.clearRect(0, 0, Canvas.getWidth(), Canvas.getHeight());
		imageResized.setImage(null);
		turnOffLights();
	}
	
	//importe puis analyse une image
	@FXML
	void importation() throws MalformedURLException {
		
		String location = NeuronalNetworks.location;
		Stage mainStage = new Stage();

		FileChooser fileChooser = new FileChooser();
		fileChooser.setTitle("Ouvrir l'image à analyser");
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
		
			double[] results = nN.forwardPropagationRAM(NeuronalNetworks.imageLecture("tmpResized"));
			turnOnLights(results);


			fileResized.delete();
		  } catch (IOException e) {
	        	e.printStackTrace();
	        } catch (ClassNotFoundException e) {
			e.printStackTrace();
		}
	}
	
	@FXML
	void init(){
		nN.extractSuccessRate();
		txSuccess.setText(nN.getSuccessRate());
	}

}
