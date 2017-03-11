package application;

import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.awt.image.RenderedImage;
import java.io.File;
import java.io.IOException;

import javafx.embed.swing.SwingFXUtils;
import javafx.fxml.FXML;
import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.control.Button;
import javafx.scene.image.WritableImage;
import javafx.scene.input.MouseEvent;
import javafx.scene.paint.Color;
import javafx.scene.paint.CycleMethod;
import javafx.scene.paint.RadialGradient;
import javafx.scene.paint.Stop;
import javafx.scene.shape.Circle;
import javafx.scene.shape.Rectangle;

import javax.imageio.ImageIO;

public class SimpleViewCtrl {
	
	@FXML Button boutonAnalyser;
	@FXML Button boutonNouveau;
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
	
	//save a png version of the canvas
    void save() {
    	String location = "/home/timoth/Documents/TSP/1A/Projet Info/";
    	//TODO Modify location
		File file = new File(location + "/images/tmp.png");
		File fileResized = new File(location + "/images/tmpResized.png");
		//TODO Modify files location
		try {
            WritableImage writableImage = new WritableImage((int)Canvas.getWidth(), (int)Canvas.getHeight());
            Canvas.snapshot(null, writableImage);
            RenderedImage renderedImage = SwingFXUtils.fromFXImage(writableImage, null);
            ImageIO.write(renderedImage, "png", file);
            
            BufferedImage resizedImage = new BufferedImage(125, 125, 1);
    		Graphics2D g = resizedImage.createGraphics();
    		g.drawImage((BufferedImage)renderedImage, 0, 0, 25, 25, null);
    		g.dispose();
    		ImageIO.write(resizedImage, "png", fileResized);
        } catch (IOException e) {
        	e.printStackTrace();
        }
    };
	
    //make a forward propagation on the canvas and return result
	@FXML
	void analyse() {
		save();
		//int result = (int)neuronalNetwork.forwardPropagation("tmpResized");
		Circle[] circles = {circle0, circle1, circle2, circle3, circle4, circle5, circle6, circle7, circle8, circle9};
		int i = (int)(Math.random()*10);
		turnOnLight(circles[i]);
	}
	
	//change color of a circle to identify it
	void turnOnLight(Circle c) {
		c.setRadius(30);
		c.setStrokeWidth(0);
		RadialGradient gradient1 = new RadialGradient(0,0,0.5,0.5,0.5,true,CycleMethod.NO_CYCLE,
				new Stop(0, Color.RED),
	            new Stop(1, Color.WHITE));
		c.setFill(gradient1);
	}
	
	//set back to normal a circle 
	void turnOffLight(Circle c) {
		c.setRadius(8);
		c.setFill(Color.RED);
		c.setStrokeWidth(1);
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
	
	//useless, was made to avoid linking line...
	@FXML
	void Release(MouseEvent event) {
		GraphicsContext gc = Canvas.getGraphicsContext2D();
        gc.beginPath();
        gc.moveTo(event.getX(), event.getY());
	}
	
	//reset canvas and circles
	@FXML
	void clear() {
		GraphicsContext gc = Canvas.getGraphicsContext2D();
		gc.clearRect(0, 0, Canvas.getWidth(), Canvas.getHeight());
		Circle[] circles = {circle0, circle1, circle2, circle3, circle4, circle5, circle6, circle7, circle8, circle9};
		for (int i=0; i<10; i++) {turnOffLight(circles[i]);}
	}

}
