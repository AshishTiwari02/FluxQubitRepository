package com.Selenium;

import org.openqa.selenium.By;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.WebElement;
import org.openqa.selenium.chrome.ChromeDriver;
import org.openqa.selenium.interactions.Actions;
import org.openqa.selenium.support.ui.Select;

public class dropDownAction {

	public static void main(String[] args) {
		String driverPath = "webdriver.chrome.driver";
		String driverURL = "D:\\Softwares\\Selenium\\chromedriver_win32\\chromedriver.exe";

		System.setProperty(driverPath, driverURL);

		/******************Handle Drop down in Web Driver****/
 
		  WebDriver driver = new ChromeDriver();
		  String baseUrl ="D:\\Practice\\HTML\\SampleDrop.html";
		  
		  try {
		 // driver.get(baseUrl);

			/*
			 * WebElement web= driver.findElement(By.xpath("/html/body/select")); Select se
			 * = new Select(web); se.selectByVisibleText("Opel");
			 * System.out.println(se.getOptions().size());
			 */
	       
	       
			/*
			 * //handle alerts pop up
			 * 
			 * driver.switchTo().alert().accept(); driver.switchTo().alert().dismiss();
			 * driver.switchTo().alert().getText();
			 * driver.switchTo().alert().sendKeys("Send name to alert box ");
			 */
	       
	       
			/*
			 * Handle a frame or popup
			 * 
			 * driver.switchTo().frame(0); driver.switchTo().frame("frameName");
			 * driver.switchTo().frame(driver.findElement(By.xpath("")));
			 */
	       
	       //Mouse actions
	       
			  
		   String MouseURL = "https://swisnl.github.io/jQuery-contextMenu/demo.html";
		   //driver.get(MouseURL);
		   
			/************************************************************
			 * **********************Mouse Related Actions***************
			 * 
			 * 
			 * Actions act = new Actions(driver);
			 * 
			 * WebElement gender =
			 * driver.findElement(By.xpath("/html/body/table/tbody/tr[5]/td[2]/input[1]"));
			 * act.moveToElement(gender).build().perform(); WebElement gender2 =
			 * driver.findElement(By.xpath("/html/body/table/tbody/tr[5]/td[2]/input[2]"));
			 * act.moveToElement(gender2).click().build().perform();
			 * 
			 * act.moveToElement(gender).moveToElement(gender2).click().build().perform(); - All Mouse action in Single line to reduce code
			 * 
			 * 		 WebElement mainButton = driver.findElement(By.xpath("/html/body/div/section/div/div/div/p/span"));
	       *
	       	*		act.contextClick(mainButton).build().perform();
	       	*		driver.findElement(By.xpath("/html/body/ul/li[3]")).click();
	       	*		System.out.println(driver.switchTo().alert().getText());
	       	*	driver.switchTo().alert().accept();
			 * 
			 * 
			 * 
			 * ***********************************************************
			 **************************************************************/
		   String dragDroplink = "http://dhtmlgoodies.com/scripts/drag-drop-custom/demo-drag-drop-3.html";
		   Actions act = new Actions(driver);
	      
	       driver.get(dragDroplink);
	       
	       WebElement source =  driver.findElement(By.id("box6"));
	       WebElement target =  driver.findElement(By.id("box106"));
	       
	       act.clickAndHold(source).moveToElement(target).release().build().perform();
	       
	      Thread.sleep(3000);
	      act.moveToElement(source).dragAndDropBy(source, 100, 100).build().perform();
	      
	      
	       
		  }
		  catch(Exception e) {
				System.out.println("Exception Descrption : " +e);
				}
			finally 
			{
				driver.close();
				driver.quit();
			}
				
			
	}

}
