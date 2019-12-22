package com.Selenium;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Set;
import java.util.concurrent.TimeUnit;

import org.openqa.selenium.By;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.WebElement;
import org.openqa.selenium.chrome.ChromeDriver;
import org.openqa.selenium.support.ui.Select;

public class demoDropDown {

	public static void main(String[] args) throws InterruptedException {

		String driverPath = "webdriver.chrome.driver";
		String driverURL = "D:\\Softwares\\Selenium\\chromedriver_win32\\chromedriver.exe";

		System.setProperty(driverPath, driverURL);

		/******************Handle Drop down in Web Driver****
		 * 
		 * 
		 * WebDriver driver = new ChromeDriver(); String baseUrl =
		 * "D:\\Practice\\HTML\\SampleDrop.html"; driver.get(baseUrl);
		 * 
		 * 
		 * WebElement web= driver.findElement(By.xpath("/html/body/select")); Select se
		 * = new Select(web);
		 * 
		 * List original = new ArrayList(); List<WebElement> option = se.getOptions();
		 * for (WebElement e:option) { original.add(e.getText()); }
		 * 
		 *//**********
			 * While validating drop down always create two separte list otherwise changes
			 * will reflect in both
			 *********/
		/*
		 * 
		 * List Temprary = new ArrayList(); List<WebElement> option22 = se.getOptions();
		 * for (WebElement e2:option) { Temprary.add(e2.getText()); }
		 * 
		 * List temp =new ArrayList(); temp = Temprary; //
		 * System.out.println("B4 : "+temp); Collections.sort(temp); //
		 * System.out.println("A4 : "+temp);
		 * 
		 * 
		 * 
		 * if(temp==original) { System.out.println("Sorted"); } else {
		 * System.out.println("Not Sorted"); }
		 * 
		 * 
		 */ 
		WebDriver driver = new ChromeDriver(); 
		String baseUrl ="https://www.amazon.in/s?k=phones&ref=nb_sb_noss_2";
		try {
		driver.get(baseUrl);
		driver.manage().timeouts().implicitlyWait(10, TimeUnit.SECONDS);
				
		String urlTitle= driver.getTitle();
		String robotCheck = "Robot Check";
		System.out.println("Current URL : "+urlTitle);
		
		if (urlTitle.equals(robotCheck))
		{
			System.out.println("Closing Robot Check Window...");
			driver.navigate().to("https://www.amazon.in/");
			System.out.println("Current URL : " + driver.getTitle());
		}

		String searchKeyword = "Phones";

		driver.findElement(By.id("twotabsearchtextbox")).sendKeys(searchKeyword);
		driver.findElement(By.className("nav-input")).click();
		System.out.println("Current URL : " +driver.getTitle());
		driver.findElement(By.xpath("//*[@id=\"search\"]/div[1]/div[2]/div/span[4]/div[1]/div[1]/div/span/div/div/div/div/div[2]/div[2]/div/div[1]/div/div/div[1]/h2/a/span")).click();
		Set <String> allURL = driver.getWindowHandles();
		for (String i :allURL)
		{	System.out.println("List of All Tabs available...");
			String tempURL = driver.switchTo().window(i).getTitle() ;
			System.out.println(tempURL);
			if (tempURL.contains("Amazon.in: Phones"))
			{
				driver.close();
			}
		}
		
		/*Handle popup windwo*/
		//driver.get("http://admin:admin@www.amazon.in");
		
		}
		
		catch(Exception e) {
			System.out.println("Exception Descrption : " +e);
			}
		finally {
			driver.quit();
		}

	}

}
