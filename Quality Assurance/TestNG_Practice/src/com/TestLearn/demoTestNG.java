package com.TestLearn;

import org.testng.annotations.BeforeMethod;

import java.util.concurrent.TimeUnit;

import org.openqa.selenium.By;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.WebElement;
import org.openqa.selenium.chrome.ChromeDriver;
import org.testng.Assert;
import org.testng.annotations.*;


//Annotations
//Priority
//Groups


public class demoTestNG {

	static WebDriver driver;
	
	@BeforeMethod
	public void setUp() throws InterruptedException
	{		
			String driverType = "webdriver.chrome.driver";
			String driverUrl = "D:\\Softwares\\Selenium\\chromedriver\\chromedriver.exe";
			System.setProperty(driverType,driverUrl);
			driver = new ChromeDriver();
			String urlPath ="https://www.google.co.in/";
			driver.get(urlPath);
			driver.manage().window().maximize();
			driver.manage().deleteAllCookies();
			driver.manage().timeouts().pageLoadTimeout(2000, TimeUnit.MILLISECONDS);
	}
	
	@Test(priority=2,groups="onGooglePage")
	public void googleTitle()
	{
		
		String title = driver.getTitle();
		System.out.println(title);
		
		/*
		 * if (title.equals("Google")) { System.out.println("Pass"); } else
		 * {System.out.println("Fail");}
		 */
		  
		  Assert.assertEquals(title, "Google","Title is not matched");
	}
	
	
	@Test(priority=1)
	public void googleImage()
	{
		
		//String title = driver.getTitle();
		boolean b = driver.findElement(By.xpath(".//*[@id=\"hplogo\"]")).isDisplayed();
		Assert.assertTrue(b);
		 
	}
	
	@Test(priority=2, groups="onGooglePage")
	public void googleSearch()
	{
		
		//String title = driver.getTitle();
		WebElement b = driver.findElement(By.xpath(".//*[@id=\"tsf\"]/div[2]/div[1]/div[1]/div/div[2]/input"));
		if( b.isDisplayed())
		{ System.out.println("Pass 3"); } else
		  {System.out.println("Fail 3");}
		 
	}
	
	
	
	@AfterMethod
	public void killTest(){
		driver.close(); 
		driver.quit();
		
	}
	
}
